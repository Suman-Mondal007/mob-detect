import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

export type DetectionSensitivity = "low" | "balanced" | "high";

export interface DetectionPreferences {
  alertThreshold: number;
  sensitivity: DetectionSensitivity;
  soundEnabled: boolean;
}

interface DetectionPreferencesContextValue {
  preferences: DetectionPreferences;
  updatePreferences: (updates: Partial<DetectionPreferences>) => void;
  setSoundEnabled: (enabled: boolean) => Promise<void>;
  playAlertTone: () => void;
}

const STORAGE_KEY = "mob-alert-detection-preferences";
const DEFAULT_PREFERENCES: DetectionPreferences = {
  alertThreshold: 0.8,
  sensitivity: "balanced",
  soundEnabled: false,
};

const DetectionPreferencesContext =
  createContext<DetectionPreferencesContextValue | null>(null);

function isDetectionPreferences(value: unknown): value is DetectionPreferences {
  if (typeof value !== "object" || value === null) return false;
  if (
    !("alertThreshold" in value) ||
    !("sensitivity" in value) ||
    !("soundEnabled" in value)
  ) {
    return false;
  }
  return (
    typeof value.alertThreshold === "number" &&
    Number.isFinite(value.alertThreshold) &&
    value.alertThreshold >= 0.3 &&
    value.alertThreshold <= 0.95 &&
    (value.sensitivity === "low" ||
      value.sensitivity === "balanced" ||
      value.sensitivity === "high") &&
    typeof value.soundEnabled === "boolean"
  );
}

function loadPreferences(): DetectionPreferences {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored === null) return DEFAULT_PREFERENCES;
    const parsed: unknown = JSON.parse(stored);
    if (isDetectionPreferences(parsed)) return parsed;
    console.warn("Ignoring invalid saved Mob-Alert detection preferences.");
  } catch (error) {
    console.warn("Could not read saved Mob-Alert detection preferences.", error);
  }
  return DEFAULT_PREFERENCES;
}

export function DetectionPreferencesProvider({
  children,
}: {
  children: ReactNode;
}) {
  const [preferences, setPreferences] =
    useState<DetectionPreferences>(loadPreferences);
  const audioContextRef = useRef<AudioContext | null>(null);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(preferences));
    } catch (error) {
      console.error("Could not save Mob-Alert detection preferences.", error);
    }
  }, [preferences]);

  const updatePreferences = useCallback(
    (updates: Partial<DetectionPreferences>) => {
      setPreferences((current) => ({ ...current, ...updates }));
    },
    [],
  );

  const setSoundEnabled = useCallback(async (enabled: boolean) => {
    if (!enabled) {
      try {
        await audioContextRef.current?.suspend();
        setPreferences((current) => ({ ...current, soundEnabled: false }));
      } catch (error) {
        throw new Error("Could not turn off alert sound.", { cause: error });
      }
      return;
    }

    try {
      const context =
        audioContextRef.current ??
        new window.AudioContext({ latencyHint: "interactive" });
      audioContextRef.current = context;
      await context.resume();
      setPreferences((current) => ({ ...current, soundEnabled: true }));
    } catch (error) {
      throw new Error("Could not enable alert sound.", { cause: error });
    }
  }, []);

  const playAlertTone = useCallback(() => {
    const context = audioContextRef.current;
    if (!context || context.state !== "running") return;
    try {
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      const now = context.currentTime;
      oscillator.type = "sine";
      oscillator.frequency.setValueAtTime(880, now);
      oscillator.frequency.setValueAtTime(660, now + 0.16);
      gain.gain.setValueAtTime(0.0001, now);
      gain.gain.exponentialRampToValueAtTime(0.18, now + 0.025);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.38);
      oscillator.connect(gain);
      gain.connect(context.destination);
      oscillator.start(now);
      oscillator.stop(now + 0.4);
    } catch (error) {
      console.error("Could not play the Mob-Alert notification tone.", error);
    }
  }, []);

  useEffect(
    () => () => {
      void audioContextRef.current?.close();
    },
    [],
  );

  const value = useMemo(
    () => ({
      preferences,
      updatePreferences,
      setSoundEnabled,
      playAlertTone,
    }),
    [preferences, updatePreferences, setSoundEnabled, playAlertTone],
  );

  return (
    <DetectionPreferencesContext.Provider value={value}>
      {children}
    </DetectionPreferencesContext.Provider>
  );
}

export function useDetectionPreferences(): DetectionPreferencesContextValue {
  const value = useContext(DetectionPreferencesContext);
  if (!value) {
    throw new Error(
      "useDetectionPreferences must be used within DetectionPreferencesProvider.",
    );
  }
  return value;
}
