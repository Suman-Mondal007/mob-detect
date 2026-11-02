import { useCallback, useEffect, useRef, useState } from "react";
import {
  getAuthToken,
  setAuthToken,
  getCameras,
  getCameraEventsWebSocketUrl,
  getHealth,
  getIncidents,
  type CameraEvent,
  type CameraStatus,
  type HealthResponse,
  type Incident,
  type ManagedCamera,
} from "../api/client";

type ConnectionState = "connecting" | "connected" | "reconnecting" | "offline";

const RECONNECT_INITIAL_MS = 500;
const RECONNECT_MAX_MS = 15_000;

function isCameraStatus(value: unknown): value is CameraStatus {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Partial<CameraStatus>;
  return (
    typeof candidate.camera_id === "string" &&
    typeof candidate.online === "boolean" &&
    (candidate.state === "NORMAL" ||
      candidate.state === "ALERT" ||
      candidate.state === "ACKNOWLEDGED" ||
      candidate.state === "COOLDOWN")
  );
}

function isCameraEvent(value: unknown): value is CameraEvent {
  if (typeof value !== "object" || value === null || !("type" in value)) {
    return false;
  }
  const candidate = value as { type?: unknown; cameras?: unknown };
  return (
    candidate.type === "heartbeat" ||
    candidate.type === "alert" ||
    (candidate.type === "camera_status" &&
      Array.isArray(candidate.cameras) &&
      candidate.cameras.every(isCameraStatus))
  );
}

function mergeCameraStatuses(
  cameras: ManagedCamera[],
  statuses: CameraStatus[],
): ManagedCamera[] {
  const byId = new Map(statuses.map((status) => [status.camera_id, status]));
  return cameras.map((camera) => ({
    ...camera,
    ...(byId.get(camera.id) ?? {}),
  }));
}

export function useCameraMonitoring() {
  const [cameras, setCameras] = useState<ManagedCamera[]>([]);
  const [connectionState, setConnectionState] =
    useState<ConnectionState>("connecting");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [incidentError, setIncidentError] = useState<string | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [eventError, setEventError] = useState<string | null>(null);
  const [lastAlert, setLastAlert] = useState<CameraEvent | null>(null);
  const [retryKey, setRetryKey] = useState(0);
  const camerasRef = useRef<ManagedCamera[]>([]);

  const publishCameras = useCallback((nextCameras: ManagedCamera[]) => {
    camerasRef.current = nextCameras;
    setCameras(nextCameras);
  }, []);

  const refreshIncidents = useCallback(async (signal?: AbortSignal) => {
    try {
      const response = await getIncidents({ signal });
      setIncidents(response.incidents);
      setIncidentError(null);
    } catch (requestError: unknown) {
      if (signal?.aborted) return;
      setIncidentError(
        requestError instanceof Error
          ? requestError.message
          : "Could not load active incidents.",
      );
    }
  }, []);

  useEffect(() => {
    let active = true;
    const controller = new AbortController();
    void getCameras(controller.signal)
      .then((cameraResponse) => {
        if (!active) return;
        publishCameras(cameraResponse.cameras.filter((camera) => camera.enabled));
        setCameraError(null);
      })
      .catch((requestError: unknown) => {
        if (!active || controller.signal.aborted) return;
        setCameraError(
          requestError instanceof Error
            ? requestError.message
            : "Could not load camera monitoring data.",
        );
      });
    void getHealth()
      .then((healthResponse) => {
        if (active) {
          setHealth(healthResponse);
          setHealthError(null);
        }
      })
      .catch((requestError: unknown) => {
        if (!active) return;
        setHealthError(
          requestError instanceof Error
            ? requestError.message
            : "Could not load model and device health.",
        );
      });
    void refreshIncidents(controller.signal);

    return () => {
      active = false;
      controller.abort();
    };
  }, [publishCameras, refreshIncidents, retryKey]);

  useEffect(() => {
    const refreshTimer = window.setInterval(() => {
      void refreshIncidents();
    }, 2500);
    return () => window.clearInterval(refreshTimer);
  }, [refreshIncidents]);

  useEffect(() => {
    let active = true;
    let retryTimer: number | undefined;
    let socket: WebSocket | undefined;
    let retryDelay = RECONNECT_INITIAL_MS;

    const connect = () => {
      if (!getAuthToken()) {
        setConnectionState("offline");
        window.location.replace("/login");
        return;
      }
      if (!active) return;
      setConnectionState((previous) =>
        previous === "connecting" ? "connecting" : "reconnecting",
      );
      let nextSocket: WebSocket;
      try {
        nextSocket = new WebSocket(getCameraEventsWebSocketUrl());
      } catch {
        setConnectionState("offline");
        retryTimer = window.setTimeout(connect, retryDelay);
        retryDelay = Math.min(retryDelay * 2, RECONNECT_MAX_MS);
        return;
      }
      socket = nextSocket;
      nextSocket.onopen = () => {
        if (!active || socket !== nextSocket) return;
        retryDelay = RECONNECT_INITIAL_MS;
        setConnectionState("connected");
        setEventError(null);
      };
      nextSocket.onmessage = (message: MessageEvent<string>) => {
        if (!active || socket !== nextSocket) return;
        let parsed: unknown;
        try {
          parsed = JSON.parse(message.data);
        } catch {
          setEventError("Received an invalid camera event from the server.");
          return;
        }
        if (!isCameraEvent(parsed)) {
          setEventError("Received an unsupported camera event from the server.");
          return;
        }
        if (parsed.type === "camera_status") {
          publishCameras(
            mergeCameraStatuses(camerasRef.current, parsed.cameras),
          );
        } else if (parsed.type === "alert") {
          setLastAlert(parsed);
          void refreshIncidents();
        }
      };
      nextSocket.onerror = () => {
        if (active && socket === nextSocket) {
          setConnectionState("reconnecting");
        }
      };
      nextSocket.onclose = (event: CloseEvent) => {
        if (!active || socket !== nextSocket) return;
        if (event.code === 4401) {
          setAuthToken(null);
          window.location.replace("/login");
          return;
        }
        setConnectionState("reconnecting");
        retryTimer = window.setTimeout(connect, retryDelay);
        retryDelay = Math.min(retryDelay * 2, RECONNECT_MAX_MS);
      };
    };

    connect();
    return () => {
      active = false;
      if (retryTimer !== undefined) window.clearTimeout(retryTimer);
      socket?.close();
    };
  }, [publishCameras, refreshIncidents]);

  const retry = useCallback(() => {
    setCameraError(null);
    setHealthError(null);
    setEventError(null);
    setRetryKey((current) => current + 1);
    setConnectionState("connecting");
  }, []);

  return {
    cameras,
    connectionState,
    health,
    incidents,
    error: cameraError ?? healthError ?? eventError,
    incidentError,
    lastAlert,
    refreshIncidents,
    retry,
  };
}
