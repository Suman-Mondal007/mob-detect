import {
  AlertTriangle,
  Bell,
  Check,
  ChevronLeft,
  ChevronRight,
  Expand,
  Eye,
  Monitor,
  Radio,
  RefreshCw,
  Volume2,
  VolumeX,
  X,
} from "lucide-react";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type MouseEvent,
  type CSSProperties,
} from "react";
import type { Incident, ManagedCamera } from "../api/client";
import {
  acknowledgeCameraAlert,
  acknowledgeIncident,
  getIncidentSnapshotUrl,
  getCameraStreamUrl,
  markIncidentFalsePositive,
} from "../api/client";
import { Toast, type ToastTone } from "../components/Toast";
import { useAlarmSound } from "../hooks/useAlarmSound";
import { useCameraMonitoring } from "../hooks/useCameraMonitoring";

type GridLayout = "1" | "4" | "9" | "16" | "auto";

const LAYOUT_OPTIONS: { id: GridLayout; label: string; title: string }[] = [
  { id: "1", label: "1", title: "Single camera" },
  { id: "4", label: "2×2", title: "2 by 2 cameras" },
  { id: "9", label: "3×3", title: "3 by 3 cameras" },
  { id: "16", label: "4×4", title: "4 by 4 cameras" },
  { id: "auto", label: "Auto-fit", title: "Auto-fit cameras to the screen" },
];

function percentage(value: number | null): number {
  if (value === null || !Number.isFinite(value)) return 0;
  return Math.round(Math.max(0, Math.min(1, value)) * 100);
}

function latencyLabel(value: number | null): string {
  return value === null || !Number.isFinite(value)
    ? "—"
    : `${Math.round(value)} ms`;
}

function useTileVisibility<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    if (!("IntersectionObserver" in window)) {
      setVisible(true);
      return;
    }

    const observer = new IntersectionObserver(
      ([entry]) => setVisible(entry.isIntersecting),
      { rootMargin: "80px" },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return { ref, visible };
}

interface CameraTileProps {
  camera: ManagedCamera;
  focused: boolean;
  gridPaused: boolean;
  streamQuality: "grid" | "full";
  onSelect: () => void;
  onClickTile: () => void;
  onFullscreen: (event: MouseEvent<HTMLElement>) => void;
}

function CameraTile({
  camera,
  focused,
  gridPaused,
  streamQuality,
  onSelect,
  onClickTile,
  onFullscreen,
}: CameraTileProps) {
  const { ref, visible } = useTileVisibility<HTMLElement>();
  const [streamFailed, setStreamFailed] = useState(false);
  const confidence = percentage(camera.smoothed_confidence);
  const isAlert = camera.state === "ALERT";
  const showStream = visible && camera.online && !streamFailed && !gridPaused;

  useEffect(() => {
    setStreamFailed(false);
  }, [camera.id, camera.online]);

  useEffect(() => {
    if (!streamFailed || !camera.online || gridPaused || !visible) return;
    const retryTimer = window.setTimeout(() => setStreamFailed(false), 3000);
    return () => window.clearTimeout(retryTimer);
  }, [camera.online, gridPaused, streamFailed, visible]);

  return (
    <article
      ref={ref}
      className={`monitor-camera-tile${isAlert ? " is-alert" : ""}${focused ? " is-focused" : ""}`}
      role="group"
      onClick={onClickTile}
      onDoubleClick={onFullscreen}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onSelect();
        }
      }}
      tabIndex={0}
      aria-roledescription="camera tile"
      aria-label={`${camera.name}, ${camera.online ? "live" : "offline"}${isAlert ? ", alert" : ""}`}
      data-camera-id={camera.id}
    >
      <div className="monitor-video-stage">
        {showStream ? (
          <img
            className="monitor-stream-image"
            src={getCameraStreamUrl(camera.id, streamQuality)}
            alt={`Live feed from ${camera.name}`}
            onError={() => setStreamFailed(true)}
            draggable={false}
          />
        ) : (
          <div className="monitor-stream-placeholder">
            {camera.online && gridPaused ? (
              <span>Grid paused while focus view is open</span>
            ) : camera.online ? (
              <span>Stream unavailable</span>
            ) : (
              <>
                <Monitor size={24} aria-hidden="true" />
                <span>Camera offline</span>
              </>
            )}
          </div>
        )}
        <div className="monitor-video-topline">
          <span className={`monitor-live-badge${camera.online ? " is-live" : ""}`}>
            <span className="monitor-live-dot" />
            {camera.online ? "LIVE" : "OFFLINE"}
          </span>
          {isAlert && (
            <span className="monitor-alert-badge">
              <AlertTriangle size={12} aria-hidden="true" />
              ALERT
            </span>
          )}
        </div>
        <button
          type="button"
          className="monitor-expand-button"
          aria-label={`Focus ${camera.name}`}
          onClick={(event) => {
            event.stopPropagation();
            onSelect();
          }}
        >
          <Expand size={15} aria-hidden="true" />
        </button>
      </div>
      <div className="monitor-tile-information">
        <div className="monitor-tile-heading">
          <div className="monitor-tile-name">
            <h2>{camera.name}</h2>
            <p>{camera.location || "Location not set"}</p>
          </div>
          <span className={`monitor-state-label${isAlert ? " is-alert" : ""}`}>
            {camera.state}
          </span>
        </div>
        <div className="monitor-tile-metrics">
          <div className="monitor-confidence">
            <div className="monitor-metric-label">
              <span>Confidence</span>
              <strong>{camera.smoothed_confidence === null ? "—" : `${confidence}%`}</strong>
            </div>
            <div
              className="monitor-confidence-track"
              role="meter"
              aria-label={`${camera.name} smoothed confidence`}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={camera.smoothed_confidence === null ? 0 : confidence}
            >
              <span
                className={isAlert ? "is-alert" : ""}
                style={{ width: `${confidence}%` }}
              />
            </div>
          </div>
          <div className="monitor-latency">
            <span>Latency</span>
            <strong>{latencyLabel(camera.inference_ms)}</strong>
          </div>
        </div>
      </div>
    </article>
  );
}

interface FocusViewProps {
  camera: ManagedCamera;
  onClose: () => void;
}

function FocusView({ camera, onClose }: FocusViewProps) {
  const confidence = percentage(camera.smoothed_confidence);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  return (
    <div
      className="monitor-focus-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section
        className="monitor-focus-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="monitor-focus-title"
      >
        <header className="monitor-focus-header">
          <div>
            <p className="monitor-focus-eyebrow">Camera focus</p>
            <h2 id="monitor-focus-title">{camera.name}</h2>
            <p>{camera.location || "Location not set"}</p>
          </div>
          <button
            type="button"
            className="monitor-focus-close"
            aria-label="Close focus view"
            onClick={onClose}
          >
            <X size={20} aria-hidden="true" />
          </button>
        </header>
        <div className="monitor-focus-stream">
          {camera.online ? (
            <img
              src={getCameraStreamUrl(camera.id, "full")}
              alt={`Full camera feed from ${camera.name}`}
            />
          ) : (
            <div className="monitor-stream-placeholder">
              <Monitor size={32} aria-hidden="true" />
              <span>Camera offline</span>
            </div>
          )}
          {camera.state === "ALERT" && (
            <span className="monitor-alert-badge monitor-focus-alert">
              <AlertTriangle size={13} aria-hidden="true" />
              ALERT
            </span>
          )}
        </div>
        <div className="monitor-focus-details">
          <div>
            <span>Connection</span>
            <strong>{camera.online ? "Live" : "Offline"}</strong>
          </div>
          <div>
            <span>Alert state</span>
            <strong>{camera.state}</strong>
          </div>
          <div>
            <span>Smoothed confidence</span>
            <strong>{camera.smoothed_confidence === null ? "—" : `${confidence}%`}</strong>
          </div>
          <div>
            <span>Inference latency</span>
            <strong>{latencyLabel(camera.inference_ms)}</strong>
          </div>
          <div>
            <span>Input rate</span>
            <strong>{camera.input_fps.toFixed(1)} FPS</strong>
          </div>
          <div>
            <span>Last frame</span>
            <strong>
              {camera.last_frame_at
                ? new Date(camera.last_frame_at).toLocaleTimeString()
                : "—"}
            </strong>
          </div>
        </div>
        <p className="monitor-focus-disclaimer">
          Prototype screening tool, not a validated safety system.
        </p>
      </section>
    </div>
  );
}

export function MonitoringPage() {
  const {
    cameras,
    connectionState,
    health,
    incidents,
    error,
    incidentError,
    lastAlert,
    refreshIncidents,
    retry,
  } =
    useCameraMonitoring();
  const [layout, setLayout] = useState<GridLayout>(() => {
    try {
      const stored = window.localStorage.getItem("mob-alert-monitor-layout");
      if (stored === "1" || stored === "4" || stored === "9" || stored === "16" || stored === "auto") {
        return stored;
      }
    } catch {
      // Fall back to default layout.
    }
    return "4";
  });
  const [page, setPage] = useState(0);
  const [focusedCameraId, setFocusedCameraId] = useState<string | null>(null);
  const [fullscreenCameraId, setFullscreenCameraId] = useState<string | null>(null);
  const [fullscreenError, setFullscreenError] = useState<string | null>(null);
  const [notificationPermission, setNotificationPermission] =
    useState<NotificationPermission | "unsupported">(() =>
      "Notification" in window ? Notification.permission : "unsupported",
    );
  const [toast, setToast] = useState<{
    title: string;
    description?: string;
    tone: ToastTone;
  } | null>(null);
  const [pendingIncidentId, setPendingIncidentId] = useState<number | null>(null);
  const focusTimerRef = useRef<number | null>(null);
  const originalTitleRef = useRef(document.title);
  const alarm = useAlarmSound(
    cameras.some((camera) => camera.state === "ALERT"),
  );
  const focusedCamera = cameras.find((camera) => camera.id === focusedCameraId);
  const activeIncidents = incidents.filter((incident) => incident.status === "new");
  const sortedCameras = useMemo(
    () =>
      [...cameras].sort((first, second) => {
        const alertOrder =
          Number(second.state === "ALERT") - Number(first.state === "ALERT");
        if (alertOrder) return alertOrder;
        const onlineOrder = Number(second.online) - Number(first.online);
        if (onlineOrder) return onlineOrder;
        let cameraOrder: string[] = [];
        try {
          const stored = window.localStorage.getItem("mob-alert-camera-order");
          if (stored) {
            const parsed = JSON.parse(stored);
            if (Array.isArray(parsed)) {
              cameraOrder = parsed.filter((item) => typeof item === "string");
            }
          }
        } catch {
          cameraOrder = [];
        }
        const orderIndex = new Map(cameraOrder.map((id, index) => [id, index]));
        const firstIndex = orderIndex.get(first.id);
        const secondIndex = orderIndex.get(second.id);
        if (firstIndex !== undefined || secondIndex !== undefined) {
          if (firstIndex === undefined) return 1;
          if (secondIndex === undefined) return -1;
          if (firstIndex !== secondIndex) return firstIndex - secondIndex;
        }
        return first.name.localeCompare(second.name);
      }),
    [cameras],
  );
  const pageSize = layout === "auto" ? sortedCameras.length : Number(layout);
  const visibleCameras =
    pageSize === 0
      ? []
      : sortedCameras.slice(page * pageSize, (page + 1) * pageSize);
  const totalPages = pageSize
    ? Math.max(1, Math.ceil(sortedCameras.length / pageSize))
    : 1;
  const onlineCount = cameras.filter((camera) => camera.online).length;
  const activeAlerts = cameras.filter((camera) => camera.state === "ALERT").length;
  const latencies = cameras
    .map((camera) => camera.inference_ms)
    .filter((latency): latency is number => latency !== null && Number.isFinite(latency));
  const averageLatency =
    latencies.length > 0
      ? latencies.reduce((total, latency) => total + latency, 0) / latencies.length
      : null;
  const healthTone =
    averageLatency === null ? "unknown" : averageLatency <= 100 ? "healthy" : "elevated";
  const deviceLabel = health
    ? `${health.device.startsWith("cuda") ? "GPU" : "CPU"} · ${health.device}`
    : "Device —";

  const showToast = useCallback(
    (title: string, tone: ToastTone, description?: string) => {
      setToast({ title, tone, description });
    },
    [],
  );

  const runIncidentAction = useCallback(
    async (incident: Incident, action: "acknowledge" | "false-positive") => {
      setPendingIncidentId(incident.id);
      try {
        if (action === "acknowledge") {
          await acknowledgeIncident(incident.id);
        } else {
          await markIncidentFalsePositive(incident.id);
        }
        await acknowledgeCameraAlert(incident.camera_id);
        await refreshIncidents();
        showToast(
          action === "acknowledge" ? "Alert acknowledged" : "Marked false positive",
          "success",
          `Camera ${cameras.find((camera) => camera.id === incident.camera_id)?.name ?? incident.camera_id}`,
        );
      } catch (actionError: unknown) {
        void refreshIncidents();
        showToast(
          "Could not complete incident action",
          "error",
          actionError instanceof Error ? actionError.message : "The request failed.",
        );
      } finally {
        setPendingIncidentId(null);
      }
    },
    [cameras, refreshIncidents, showToast],
  );

  const acknowledgeCamera = useCallback(
    async (cameraId: string) => {
      const incident = activeIncidents.find((item) => item.camera_id === cameraId);
      if (incident) {
        await runIncidentAction(incident, "acknowledge");
        return;
      }
      try {
        await acknowledgeCameraAlert(cameraId);
        showToast("Alert acknowledged", "success");
      } catch (actionError: unknown) {
        showToast(
          "Could not acknowledge alert",
          "error",
          actionError instanceof Error ? actionError.message : "The request failed.",
        );
      }
    },
    [activeIncidents, runIncidentAction, showToast],
  );

  useEffect(() => {
    setPage(0);
  }, [layout]);

  useEffect(() => {
    try {
      window.localStorage.setItem("mob-alert-monitor-layout", layout);
    } catch {
      // Ignore storage failures.
    }
  }, [layout]);

  useEffect(() => {
    if (activeAlerts > 0) setPage(0);
  }, [activeAlerts]);

  useEffect(() => {
    const handleFullscreenChange = () => {
      if (!document.fullscreenElement) setFullscreenCameraId(null);
    };
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

  useEffect(() => {
    if (focusedCameraId && !cameras.some((camera) => camera.id === focusedCameraId)) {
      setFocusedCameraId(null);
    }
  }, [cameras, focusedCameraId]);

  const closeFocus = useCallback(() => setFocusedCameraId(null), []);

  const selectCamera = useCallback((cameraId: string) => {
    if (focusTimerRef.current !== null) {
      window.clearTimeout(focusTimerRef.current);
      focusTimerRef.current = null;
    }
    setFocusedCameraId(cameraId);
  }, []);

  const handleTileClick = useCallback((cameraId: string) => {
    if (focusTimerRef.current !== null) {
      window.clearTimeout(focusTimerRef.current);
    }
    focusTimerRef.current = window.setTimeout(() => {
      setFocusedCameraId(cameraId);
      focusTimerRef.current = null;
    }, 220);
  }, []);

  useEffect(
    () => () => {
      if (focusTimerRef.current !== null) {
        window.clearTimeout(focusTimerRef.current);
      }
    },
    [],
  );

  const handleFullscreen = useCallback(
    (event: MouseEvent<HTMLElement>) => {
      event.stopPropagation();
      const cameraId = event.currentTarget.dataset.cameraId ?? null;
      if (focusTimerRef.current !== null) {
        window.clearTimeout(focusTimerRef.current);
        focusTimerRef.current = null;
      }
      setFocusedCameraId(null);
      setFullscreenCameraId(cameraId);
      setFullscreenError(null);
      if (document.fullscreenElement) return;
      const request = event.currentTarget.requestFullscreen?.();
      if (!request) {
        setFullscreenCameraId(null);
        setFullscreenError("Fullscreen is unavailable in this browser.");
        return;
      }
      void request.catch(() => {
          setFullscreenCameraId(null);
          setFullscreenError("Fullscreen is unavailable in this browser.");
      });
    },
    [],
  );

  const requestCameraFullscreen = useCallback(
    async (cameraId: string) => {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
        return;
      }
      const tile = Array.from(
        document.querySelectorAll<HTMLElement>("[data-camera-id]"),
      ).find((element) => element.dataset.cameraId === cameraId);
      if (!tile?.requestFullscreen) {
        setFullscreenError("Fullscreen is unavailable for this camera.");
        return;
      }
      setFullscreenCameraId(cameraId);
      setFullscreenError(null);
      try {
        await tile.requestFullscreen();
      } catch {
        setFullscreenCameraId(null);
        setFullscreenError("Fullscreen is unavailable in this browser.");
      }
    },
    [],
  );

  useEffect(() => {
    if (!lastAlert || lastAlert.type !== "alert") return;
    const cameraName =
      cameras.find((camera) => camera.id === lastAlert.camera_id)?.name ??
      lastAlert.camera_id;
    showToast("Violence alert", "error", `${cameraName} requires attention.`);
    if (notificationPermission === "granted" && "Notification" in window) {
      try {
        new Notification("Mob-Alert: violence alert", {
          body: `${cameraName} requires attention.`,
          tag: `mob-alert-${lastAlert.camera_id}`,
        });
      } catch (notificationError: unknown) {
        showToast(
          "Could not show browser notification",
          "error",
          notificationError instanceof Error
            ? notificationError.message
            : "The browser rejected the notification.",
        );
      }
    }
  }, [cameras, lastAlert, notificationPermission, showToast]);

  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(null), 5000);
    return () => window.clearTimeout(timer);
  }, [toast]);

  useEffect(() => {
    let timer: number | undefined;
    let flashing = false;
    const syncTitle = () => {
      if (timer !== undefined) {
        window.clearInterval(timer);
        timer = undefined;
      }
      if (activeAlerts > 0 && document.hidden) {
        timer = window.setInterval(() => {
          flashing = !flashing;
          document.title = flashing
            ? `ALERT (${activeAlerts}) - Mob-Alert`
            : originalTitleRef.current;
        }, 800);
      } else {
        document.title = originalTitleRef.current;
      }
    };
    const restoreTitle = () => {
      document.title = originalTitleRef.current;
    };
    syncTitle();
    document.addEventListener("visibilitychange", syncTitle);
    return () => {
      if (timer !== undefined) window.clearInterval(timer);
      document.removeEventListener("visibilitychange", syncTitle);
      restoreTitle();
    };
  }, [activeAlerts]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target;
      if (
        event.altKey ||
        event.ctrlKey ||
        event.metaKey ||
        (target instanceof HTMLElement &&
          (target.isContentEditable ||
            ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)))
      ) {
        return;
      }
      const key = event.key.toLowerCase();
      if (key === "a") {
        const camera =
          cameras.find(
            (item) => item.id === focusedCameraId && item.state === "ALERT",
          ) ?? cameras.find((item) => item.state === "ALERT");
        if (camera) {
          event.preventDefault();
          void acknowledgeCamera(camera.id);
        }
      } else if (key === "m") {
        event.preventDefault();
        alarm.toggleMute();
      } else if (key === "f") {
        const cameraId =
          focusedCameraId ??
          cameras.find((item) => item.state === "ALERT")?.id ??
          cameras[0]?.id;
        if (cameraId) {
          event.preventDefault();
          void requestCameraFullscreen(cameraId);
        }
      } else if (["1", "2", "3", "4"].includes(key)) {
        event.preventDefault();
        const shortcutLayouts: Record<string, GridLayout> = {
          "1": "1",
          "2": "4",
          "3": "9",
          "4": "16",
        };
        setLayout(shortcutLayouts[key]);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [
    acknowledgeCamera,
    alarm.toggleMute,
    cameras,
    focusedCameraId,
    requestCameraFullscreen,
  ]);

  const enableNotifications = async () => {
    if (!("Notification" in window)) {
      setNotificationPermission("unsupported");
      return;
    }
    try {
      setNotificationPermission(await Notification.requestPermission());
    } catch (permissionError: unknown) {
      showToast(
        "Could not enable notifications",
        "error",
        permissionError instanceof Error
          ? permissionError.message
          : "The browser rejected the permission request.",
      );
    }
  };

  const firstItem = visibleCameras.length ? page * pageSize + 1 : 0;
  const lastItem = page * pageSize + visibleCameras.length;
  const connectionLabel: Record<typeof connectionState, string> = {
    connecting: "Connecting",
    connected: "Connected",
    reconnecting: "Reconnecting",
    offline: "Offline",
  };

  return (
    <div className="monitoring-page">
      <div className="monitoring-main">
        <header className="monitoring-heading">
          <div>
            <p className="monitoring-eyebrow">
              <span className="monitoring-eyebrow-dot" />
              MOB-ALERT · OPERATIONS
            </p>
            <h1>Monitoring</h1>
            <p className="monitoring-heading-copy">
              Live camera wall and real-time screening status.
            </p>
          </div>
          <div className={`monitor-connection-badge is-${connectionState}`}>
            <span />
            {connectionLabel[connectionState]}
          </div>
        </header>

        <section className="monitoring-statusbar" aria-label="System status">
          <div className="monitoring-status-item">
            <span className="monitoring-status-icon is-green">
              <Radio size={16} aria-hidden="true" />
            </span>
            <div>
              <strong>{onlineCount}<span> / {cameras.length}</span></strong>
              <small>Cameras online · {cameras.length - onlineCount} offline</small>
            </div>
          </div>
          <div className="monitoring-status-divider" />
          <div className="monitoring-status-item">
            <span className={`monitoring-status-icon${activeAlerts ? " is-red" : " is-muted"}`}>
              <AlertTriangle size={16} aria-hidden="true" />
            </span>
            <div>
              <strong className={activeAlerts ? "has-alert" : ""}>{activeAlerts}</strong>
              <small>Active alerts</small>
            </div>
          </div>
          <div className="monitoring-status-divider" />
          <div className="monitoring-status-item monitor-system-health">
            <span className={`monitoring-status-icon is-${healthTone}`}>
              <Monitor size={16} aria-hidden="true" />
            </span>
            <div>
              <strong>{deviceLabel}</strong>
              <small>
                {health?.model_loaded
                  ? "Inference device · Model ready"
                  : "Model status unavailable"}
              </small>
            </div>
          </div>
          <div className="monitoring-status-divider" />
          <div className="monitoring-status-item monitor-latency-health">
            <span className={`monitor-health-dot is-${healthTone}`} />
            <div>
              <strong>{latencyLabel(averageLatency)}</strong>
              <small>
                {averageLatency === null
                  ? "Inference latency"
                  : healthTone === "healthy"
                    ? "Latency healthy"
                    : "Latency elevated"}
              </small>
            </div>
          </div>
          <button
            type="button"
            className="monitor-refresh-button"
            onClick={retry}
            aria-label="Refresh camera list and health"
            title="Refresh"
          >
            <RefreshCw size={15} aria-hidden="true" />
          </button>
        </section>

        <section className="monitoring-alarm-controls" aria-label="Alarm controls">
          <div className="monitoring-alarm-actions">
            {!alarm.enabled ? (
              <button
                type="button"
                className="monitor-control-button is-primary"
                onClick={() => {
                  void alarm.enable().catch((enableError: unknown) =>
                    showToast(
                      "Could not enable alarm sound",
                      "error",
                      enableError instanceof Error
                        ? enableError.message
                        : "Web Audio could not be started.",
                    ),
                  );
                }}
              >
                <Volume2 size={16} aria-hidden="true" />
                Enable alarm sound
              </button>
            ) : (
              <button
                type="button"
                className="monitor-control-button"
                onClick={alarm.toggleMute}
                aria-pressed={alarm.muted}
              >
                {alarm.muted ? (
                  <VolumeX size={16} aria-hidden="true" />
                ) : (
                  <Volume2 size={16} aria-hidden="true" />
                )}
                {alarm.muted ? "Unmute alarm" : "Mute alarm"}
              </button>
            )}
            <label className="monitor-volume-control">
              <span>Volume</span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={alarm.volume}
                aria-label="Alarm volume"
                onChange={(event) => alarm.setVolume(Number(event.currentTarget.value))}
              />
            </label>
            {notificationPermission !== "granted" &&
              notificationPermission !== "unsupported" && (
                <button
                  type="button"
                  className="monitor-control-button"
                  onClick={() => void enableNotifications()}
                  disabled={notificationPermission === "denied"}
                >
                  <Bell size={15} aria-hidden="true" />
                  {notificationPermission === "denied"
                    ? "Notifications blocked"
                    : "Enable notifications"}
                </button>
              )}
            {notificationPermission === "granted" && (
              <span className="monitor-control-button is-status">
                <Bell size={15} aria-hidden="true" />
                Notifications enabled
              </span>
            )}
            {notificationPermission === "unsupported" && (
              <span className="monitor-control-button is-status">
                <Bell size={15} aria-hidden="true" />
                Notifications unavailable
              </span>
            )}
          </div>
          {(!alarm.enabled || alarm.muted || alarm.volume === 0) && (
            <p className="monitor-muted-warning" role="status">
              <VolumeX size={14} aria-hidden="true" />
          {alarm.muted
            ? "Alarm is muted."
            : alarm.volume === 0
              ? "Alarm volume is set to zero."
              : "Alarm sound is not enabled in this browser."}
            </p>
          )}
          {notificationPermission === "denied" && (
            <p className="monitor-notification-warning" role="status">
              Browser notifications are blocked. Allow them in your browser settings.
            </p>
          )}
          <p className="monitoring-shortcut-hint">
            Shortcuts: A acknowledge · M mute · F fullscreen · 1–4 layouts
          </p>
        </section>

        <aside className="monitoring-incident-panel" aria-label="Active incidents">
          <div className="monitoring-incident-heading">
            <div>
              <h2>Active incidents</h2>
              <span>{activeIncidents.length} awaiting review</span>
            </div>
            <button
              type="button"
              className="monitor-control-button"
              onClick={() => void refreshIncidents()}
              aria-label="Refresh incidents"
            >
              <RefreshCw size={14} aria-hidden="true" />
              Refresh
            </button>
          </div>
          {incidentError && (
            <p className="monitor-incident-error" role="alert">{incidentError}</p>
          )}
          {activeIncidents.length === 0 ? (
            <p className="monitoring-incident-empty">No incidents are awaiting review.</p>
          ) : (
            <div className="monitoring-incident-list">
              {activeIncidents.map((incident) => {
                const camera = cameras.find((item) => item.id === incident.camera_id);
                return (
                  <article className="monitoring-incident-card" key={incident.id}>
                    {incident.snapshot_url && (
                      <img
                        className="monitoring-incident-snapshot"
                        src={getIncidentSnapshotUrl(incident.snapshot_url)}
                        alt={`Incident snapshot from ${camera?.name ?? incident.camera_id}`}
                        loading="lazy"
                      />
                    )}
                    <div className="monitoring-incident-info">
                      <strong>{camera?.name ?? incident.camera_id}</strong>
                      <span>
                        {camera?.location ?? "Unknown location"} ·{" "}
                        {new Date(incident.start_time).toLocaleString()}
                      </span>
                      <span>{percentage(incident.peak_confidence)}% peak confidence</span>
                    </div>
                    <div className="monitoring-incident-actions">
                      <button
                        type="button"
                        onClick={() => camera && selectCamera(camera.id)}
                        disabled={!camera}
                        aria-label={`View ${camera?.name ?? incident.camera_id}`}
                      >
                        <Eye size={14} aria-hidden="true" /> View
                      </button>
                      <button
                        type="button"
                        onClick={() => void runIncidentAction(incident, "acknowledge")}
                        disabled={pendingIncidentId === incident.id}
                      >
                        <Check size={14} aria-hidden="true" /> Acknowledge
                      </button>
                      <button
                        type="button"
                        onClick={() => void runIncidentAction(incident, "false-positive")}
                        disabled={pendingIncidentId === incident.id}
                      >
                        False Positive
                      </button>
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </aside>

        {error && (
          <div className="monitoring-error" role="status">
            <span>{error}</span>
            <button type="button" onClick={retry}>Retry</button>
          </div>
        )}
        {fullscreenError && (
          <div className="monitoring-error" role="status">
            <span>{fullscreenError}</span>
            <button type="button" onClick={() => setFullscreenError(null)}>Dismiss</button>
          </div>
        )}
        {lastAlert?.type === "alert" && (
          <div className="monitoring-alert-notice" role="status" aria-live="polite">
            <AlertTriangle size={15} aria-hidden="true" />
            Alert on {cameras.find((camera) => camera.id === lastAlert.camera_id)?.name ?? lastAlert.camera_id}
            {lastAlert.smoothed_confidence !== null &&
              ` · ${percentage(lastAlert.smoothed_confidence)}% confidence`}
          </div>
        )}

        <div className="monitoring-grid-toolbar">
          <div className="monitoring-grid-title">
            <h2>Camera wall</h2>
            <span>{cameras.length} {cameras.length === 1 ? "camera" : "cameras"}</span>
          </div>
          <div className="monitoring-grid-controls">
            <div className="monitor-layout-picker" role="group" aria-label="Camera layout">
              {LAYOUT_OPTIONS.map((option) => (
                <button
                  key={option.id}
                  type="button"
                  className={layout === option.id ? "is-selected" : ""}
                  aria-pressed={layout === option.id}
                  title={option.title}
                  onClick={() => setLayout(option.id)}
                >
                  {option.label}
                </button>
              ))}
            </div>
            <div className="monitor-grid-pagination">
              <span>{firstItem}–{lastItem} of {cameras.length}</span>
              <button
                type="button"
                aria-label="Previous camera page"
                disabled={page === 0}
                onClick={() => setPage((current) => Math.max(0, current - 1))}
              >
                <ChevronLeft size={16} aria-hidden="true" />
              </button>
              <button
                type="button"
                aria-label="Next camera page"
                disabled={page + 1 >= totalPages}
                onClick={() => setPage((current) => Math.min(totalPages - 1, current + 1))}
              >
                <ChevronRight size={16} aria-hidden="true" />
              </button>
            </div>
          </div>
        </div>

        {cameras.length === 0 ? (
          <div className="monitoring-empty">
            <Monitor size={28} aria-hidden="true" />
            <h2>No enabled cameras</h2>
            <p>Add or enable cameras through the camera management API to populate this wall.</p>
          </div>
        ) : (
          <section
            className={`monitoring-camera-grid${layout === "auto" ? " is-auto-fit" : ""}${layout === "1" ? " is-single-layout" : ""}`}
            style={
              layout === "auto"
                ? undefined
                : ({
                    "--monitor-columns":
                      layout === "1" ? 1 : Math.sqrt(Number(layout)),
                  } as CSSProperties)
            }
            aria-label="Live camera tiles"
          >
            {visibleCameras.map((camera) => (
              <CameraTile
                key={camera.id}
                camera={camera}
                focused={focusedCameraId === camera.id}
                gridPaused={Boolean(focusedCameraId)}
                streamQuality={fullscreenCameraId === camera.id ? "full" : "grid"}
                onSelect={() => selectCamera(camera.id)}
                onClickTile={() => handleTileClick(camera.id)}
                onFullscreen={handleFullscreen}
              />
            ))}
          </section>
        )}
        <p className="monitoring-disclaimer">
          Prototype screening tool, not a validated safety system.
        </p>
        {toast && (
          <div className="monitoring-toast" aria-live="polite">
            <Toast
              title={toast.title}
              description={toast.description}
              tone={toast.tone}
              onDismiss={() => setToast(null)}
            />
          </div>
        )}
      </div>

      {focusedCamera && (
        <FocusView camera={focusedCamera} onClose={closeFocus} />
      )}
    </div>
  );
}
