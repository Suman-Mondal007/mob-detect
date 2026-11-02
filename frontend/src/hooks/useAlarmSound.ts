import { useCallback, useEffect, useRef, useState } from "react";

const SOUND_MUTED_KEY = "mob-alert-alarm-muted";
const SOUND_VOLUME_KEY = "mob-alert-alarm-volume";
const ALARM_REPEAT_MS = 1100;

function readStoredBoolean(key: string, fallback: boolean): boolean {
  const value = window.localStorage.getItem(key);
  return value === null ? fallback : value === "true";
}

function readStoredVolume(): number {
  const value = Number(window.localStorage.getItem(SOUND_VOLUME_KEY));
  return Number.isFinite(value) && value >= 0 && value <= 1 ? value : 0.35;
}

export function useAlarmSound(active: boolean) {
  const [enabled, setEnabled] = useState(false);
  const [muted, setMuted] = useState(() =>
    readStoredBoolean(SOUND_MUTED_KEY, false),
  );
  const [volume, setVolumeState] = useState(readStoredVolume);
  const contextRef = useRef<AudioContext | null>(null);

  const enable = useCallback(async () => {
    const AudioContextConstructor =
      window.AudioContext ??
      (window as typeof window & { webkitAudioContext?: typeof AudioContext })
        .webkitAudioContext;
    if (!AudioContextConstructor) {
      throw new Error("Web Audio is not supported by this browser.");
    }

    const context = contextRef.current ?? new AudioContextConstructor();
    contextRef.current = context;
    if (context.state === "suspended") await context.resume();
    setEnabled(true);
  }, []);

  const toggleMute = useCallback(() => {
    setMuted((current) => {
      const next = !current;
      window.localStorage.setItem(SOUND_MUTED_KEY, String(next));
      return next;
    });
  }, []);

  const setVolume = useCallback((nextVolume: number) => {
    const boundedVolume = Math.max(0, Math.min(1, nextVolume));
    setVolumeState(boundedVolume);
    window.localStorage.setItem(SOUND_VOLUME_KEY, String(boundedVolume));
  }, []);

  useEffect(() => {
    if (!active || !enabled || muted || volume === 0 || !contextRef.current) {
      return;
    }

    const context = contextRef.current;
    const playBeep = (frequency: number, startAt: number, duration: number) => {
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      oscillator.type = "sine";
      oscillator.frequency.setValueAtTime(frequency, startAt);
      gain.gain.setValueAtTime(0.0001, startAt);
      gain.gain.exponentialRampToValueAtTime(volume * 0.18, startAt + 0.015);
      gain.gain.setValueAtTime(volume * 0.18, startAt + duration - 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, startAt + duration);
      oscillator.connect(gain);
      gain.connect(context.destination);
      oscillator.start(startAt);
      oscillator.stop(startAt + duration);
    };

    const playCycle = () => {
      const startAt = context.currentTime + 0.01;
      playBeep(880, startAt, 0.16);
      playBeep(660, startAt + 0.23, 0.18);
    };

    playCycle();
    const repeatTimer = window.setInterval(playCycle, ALARM_REPEAT_MS);
    return () => window.clearInterval(repeatTimer);
  }, [active, enabled, muted, volume]);

  useEffect(
    () => () => {
      const context = contextRef.current;
      contextRef.current = null;
      if (context && context.state !== "closed") {
        void context.close();
      }
    },
    [],
  );

  return {
    enabled,
    muted,
    volume,
    enable,
    toggleMute,
    setVolume,
  };
}
