import { useCallback, useEffect, useRef, useState } from "react";
import {
  getAlerts,
  getLiveWebSocketUrl,
  type LiveAlert,
} from "../api/client";
import { useDetectionPreferences } from "../components/DetectionPreferences";

type ConnectionStatus =
  | "disconnected"
  | "requesting"
  | "connecting"
  | "live"
  | "reconnecting"
  | "error";

export interface LivePrediction {
  label: string;
  confidence: number;
  smoothed_confidence: number;
  alert: boolean;
  inference_ms: number;
  timestamp: string;
}

export type LiveAlertItem = Pick<
  LiveAlert,
  "timestamp" | "label" | "confidence" | "smoothed_confidence"
>;

interface LiveMessage extends Partial<LivePrediction> {
  error?: string;
}

const FRAME_INTERVAL_MS = 250;
const MAX_FRAME_WIDTH = 320;
const MAX_SOCKET_BUFFER_BYTES = 1_048_576;
const RECONNECT_DELAYS_MS = [500, 1000, 2000, 4000, 8000];
const MAX_DISPLAYED_ALERTS = 20;

function isLiveMessage(value: unknown): value is LiveMessage {
  return typeof value === "object" && value !== null;
}

function mediaErrorMessage(error: unknown): string {
  if (!(error instanceof DOMException)) {
    return error instanceof Error ? error.message : "Could not access the camera.";
  }
  switch (error.name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Camera permission was denied. Allow camera access in your browser settings and try again.";
    case "NotFoundError":
    case "DevicesNotFoundError":
      return "No camera was found. Connect a camera and try again.";
    case "NotReadableError":
    case "TrackStartError":
      return "The camera is already in use or unavailable. Close other camera apps and try again.";
    case "OverconstrainedError":
      return "The selected camera is unavailable. Choose another camera and try again.";
    default:
      return `Could not start the camera: ${error.message}`;
  }
}

function isPrediction(value: LiveMessage): value is LiveMessage & LivePrediction {
  return (
    typeof value.label === "string" &&
    typeof value.confidence === "number" &&
    typeof value.smoothed_confidence === "number" &&
    typeof value.alert === "boolean" &&
    typeof value.inference_ms === "number" &&
    typeof value.timestamp === "string"
  );
}

export function useLiveDetection() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const captureTimerRef = useRef<number | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const reconnectAttemptRef = useRef(0);
  const capturePendingRef = useRef(false);
  const generationRef = useRef(0);
  const mountedRef = useRef(false);
  const previousAlertRef = useRef(false);
  const resultCountRef = useRef(0);
  const resultWindowStartRef = useRef(0);
  const connectWebSocketRef = useRef<
    (generation: number, reconnecting: boolean) => void
  >(() => undefined);

  const [status, setStatus] = useState<ConnectionStatus>("disconnected");
  const [cameraActive, setCameraActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState("");
  const [prediction, setPrediction] = useState<LivePrediction | null>(null);
  const [fps, setFps] = useState(0);
  const [alerts, setAlerts] = useState<LiveAlertItem[]>([]);
  const [alertsError, setAlertsError] = useState<string | null>(null);
  const { preferences, setSoundEnabled, playAlertTone } =
    useDetectionPreferences();
  const preferencesRef = useRef(preferences);
  preferencesRef.current = preferences;
  const soundEnabled = preferences.soundEnabled;

  const refreshCameras = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) {
      setError("Camera access requires a secure page (HTTPS or localhost).");
      return;
    }
    try {
      const availableDevices = await navigator.mediaDevices.enumerateDevices();
      const cameras = availableDevices.filter(
        (device) => device.kind === "videoinput",
      );
      setDevices(cameras);
      setSelectedCameraId((current) => {
        if (current && cameras.some((camera) => camera.deviceId === current)) {
          return current;
        }
        return cameras[0]?.deviceId ?? "";
      });
    } catch (cameraError) {
      setError(mediaErrorMessage(cameraError));
    }
  }, []);

  const stopCaptureTimer = useCallback(() => {
    if (captureTimerRef.current !== null) {
      window.clearInterval(captureTimerRef.current);
      captureTimerRef.current = null;
    }
  }, []);

  const captureAndSend = useCallback((socket: WebSocket) => {
    const video = videoRef.current;
    if (
      !video ||
      socket.readyState !== WebSocket.OPEN ||
      socket.bufferedAmount > MAX_SOCKET_BUFFER_BYTES ||
      video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
      capturePendingRef.current
    ) {
      return;
    }

    const sourceWidth = video.videoWidth;
    const sourceHeight = video.videoHeight;
    if (sourceWidth <= 0 || sourceHeight <= 0) return;

    const canvas =
      captureCanvasRef.current ??
      (captureCanvasRef.current = document.createElement("canvas"));
    const scale = Math.min(1, MAX_FRAME_WIDTH / sourceWidth);
    canvas.width = Math.max(1, Math.round(sourceWidth * scale));
    canvas.height = Math.max(1, Math.round(sourceHeight * scale));
    const context = canvas.getContext("2d");
    if (!context) {
      setError("Could not prepare a frame for live analysis.");
      return;
    }
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    capturePendingRef.current = true;
    canvas.toBlob(
      (blob) => {
        capturePendingRef.current = false;
        if (
          !blob ||
          socketRef.current !== socket ||
          socket.readyState !== WebSocket.OPEN
        ) {
          return;
        }
        if (blob.size > MAX_SOCKET_BUFFER_BYTES) {
          setError("Camera frame exceeds the live upload limit.");
          return;
        }
        socket.send(blob);
      },
      "image/jpeg",
      0.72,
    );
  }, []);

  const startCaptureTimer = useCallback(
    (socket: WebSocket) => {
      stopCaptureTimer();
      captureAndSend(socket);
      captureTimerRef.current = window.setInterval(
        () => captureAndSend(socket),
        FRAME_INTERVAL_MS,
      );
    },
    [captureAndSend, stopCaptureTimer],
  );

  const scheduleReconnect = useCallback(
    (generation: number) => {
      if (generation !== generationRef.current || !mountedRef.current) return;
      setStatus("reconnecting");
      stopCaptureTimer();
      const attempt = Math.min(
        reconnectAttemptRef.current,
        RECONNECT_DELAYS_MS.length - 1,
      );
      const delay = RECONNECT_DELAYS_MS[attempt];
      reconnectAttemptRef.current += 1;
      reconnectTimerRef.current = window.setTimeout(() => {
        reconnectTimerRef.current = null;
        if (generation === generationRef.current) {
          connectWebSocketRef.current(generation, true);
        }
      }, delay);
    },
    [stopCaptureTimer],
  );

  const connectWebSocket = useCallback(
    (generation: number, reconnecting: boolean) => {
      if (generation !== generationRef.current || !mountedRef.current) return;
      setStatus(reconnecting ? "reconnecting" : "connecting");

      let socket: WebSocket;
      try {
        socket = new WebSocket(getLiveWebSocketUrl());
      } catch (socketError) {
        setError(
          socketError instanceof Error
            ? socketError.message
            : "Could not connect to the live inference service.",
        );
        scheduleReconnect(generation);
        return;
      }

      socket.binaryType = "arraybuffer";
      socketRef.current = socket;
      socket.onopen = () => {
        if (generation !== generationRef.current) {
          socket.close();
          return;
        }
        try {
          socket.send(
            JSON.stringify({
              threshold: preferencesRef.current.alertThreshold,
              sensitivity: preferencesRef.current.sensitivity,
            }),
          );
        } catch (settingsError) {
          setError(
            settingsError instanceof Error
              ? `Could not apply live alert settings: ${settingsError.message}`
              : "Could not apply live alert settings.",
          );
          socket.close();
          return;
        }
        setError(null);
        setStatus("live");
        startCaptureTimer(socket);
      };
      socket.onmessage = (event: MessageEvent<string>) => {
        let payload: unknown;
        try {
          payload = JSON.parse(event.data);
        } catch {
          setError("The live inference service returned an invalid response.");
          return;
        }
        if (!isLiveMessage(payload)) return;
        if (payload.error) {
          setError(payload.error);
          return;
        }
        if (!isPrediction(payload)) return;

        reconnectAttemptRef.current = 0;
        setPrediction(payload);
        const now = performance.now();
        if (resultWindowStartRef.current === 0) {
          resultWindowStartRef.current = now;
        }
        resultCountRef.current += 1;
        const elapsed = now - resultWindowStartRef.current;
        if (elapsed >= 1000) {
          setFps((resultCountRef.current * 1000) / elapsed);
          resultCountRef.current = 0;
          resultWindowStartRef.current = now;
        }

        if (payload.alert && !previousAlertRef.current) {
          setAlerts((current) =>
            [
              {
                timestamp: payload.timestamp,
                label: payload.label,
                confidence: payload.confidence,
                smoothed_confidence: payload.smoothed_confidence,
              },
              ...current,
            ].slice(0, MAX_DISPLAYED_ALERTS),
          );
          if (preferencesRef.current.soundEnabled) playAlertTone();
        }
        previousAlertRef.current = payload.alert;
      };
      socket.onerror = () => {
        setError("Connection to live inference was interrupted.");
      };
      socket.onclose = () => {
        if (socketRef.current === socket) socketRef.current = null;
        if (generation === generationRef.current) {
          scheduleReconnect(generation);
        }
      };
    },
    [playAlertTone, scheduleReconnect, startCaptureTimer],
  );
  connectWebSocketRef.current = connectWebSocket;

  useEffect(() => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    try {
      socket.send(
        JSON.stringify({
          threshold: preferences.alertThreshold,
          sensitivity: preferences.sensitivity,
        }),
      );
    } catch (settingsError) {
      setError(
        settingsError instanceof Error
          ? `Could not apply live alert settings: ${settingsError.message}`
          : "Could not apply live alert settings.",
      );
    }
  }, [preferences.alertThreshold, preferences.sensitivity]);

  const stop = useCallback(() => {
    generationRef.current += 1;
    if (captureTimerRef.current !== null) {
      window.clearInterval(captureTimerRef.current);
      captureTimerRef.current = null;
    }
    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    const socket = socketRef.current;
    socketRef.current = null;
    if (socket && socket.readyState < WebSocket.CLOSING) {
      socket.onclose = null;
      socket.close(1000, "Live session stopped.");
    }
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    previousAlertRef.current = false;
    resultCountRef.current = 0;
    resultWindowStartRef.current = 0;
    if (mountedRef.current) {
      setCameraActive(false);
      setPrediction(null);
      setFps(0);
      setError(null);
      setStatus("disconnected");
    }
  }, []);

  const start = useCallback(async () => {
    if (status === "requesting" || status === "connecting" || status === "live") {
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatus("error");
      setError("Camera access requires a secure page (HTTPS or localhost).");
      return;
    }

    const generation = generationRef.current + 1;
    generationRef.current = generation;
    setStatus("requesting");
    setError(null);
    setPrediction(null);
    setFps(0);

    if (preferencesRef.current.soundEnabled) {
      void setSoundEnabled(true).catch((audioError: unknown) => {
        setError(
          audioError instanceof Error
            ? audioError.message
            : "Could not enable alert sound.",
        );
      });
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: selectedCameraId
          ? { deviceId: { exact: selectedCameraId } }
          : true,
      });
      if (!mountedRef.current || generation !== generationRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      streamRef.current = stream;
      setStatus("connecting");
      setCameraActive(true);
      stream.getVideoTracks().forEach((track) => {
        track.addEventListener(
          "ended",
          () => {
            if (generation !== generationRef.current || !mountedRef.current) {
              return;
            }
            stop();
            setStatus("error");
            setError("The camera disconnected. Reconnect it and try again.");
          },
          { once: true },
        );
      });
      await refreshCameras();
      if (generation === generationRef.current) {
        connectWebSocket(generation, false);
      }
    } catch (cameraError) {
      if (generation !== generationRef.current) return;
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      setCameraActive(false);
      setStatus("error");
      setError(mediaErrorMessage(cameraError));
    }
  }, [
    connectWebSocket,
    refreshCameras,
    selectedCameraId,
    setSoundEnabled,
    status,
    stop,
  ]);

  const toggleSound = useCallback(async () => {
    try {
      await setSoundEnabled(!soundEnabled);
    } catch (audioError) {
      setError(
        audioError instanceof Error
          ? audioError.message
          : "Could not change alert sound.",
      );
    }
  }, [setSoundEnabled, soundEnabled]);

  useEffect(() => {
    mountedRef.current = true;
    void refreshCameras();
    void getAlerts()
      .then(({ alerts: loadedAlerts }) => {
        if (mountedRef.current) {
          setAlerts(
            loadedAlerts
              .slice()
              .sort((left, right) =>
                right.timestamp.localeCompare(left.timestamp),
              )
              .slice(0, MAX_DISPLAYED_ALERTS)
              .map((alert) => ({
            timestamp: alert.timestamp,
            label: alert.label,
            confidence: alert.confidence,
            smoothed_confidence: alert.smoothed_confidence,
              })),
          );
        }
      })
      .catch((alertError: unknown) => {
        if (mountedRef.current) {
          setAlertsError(
            alertError instanceof Error
              ? alertError.message
              : "Could not load previous alerts.",
          );
        }
      });
    navigator.mediaDevices?.addEventListener("devicechange", refreshCameras);
    return () => {
      mountedRef.current = false;
      stop();
      navigator.mediaDevices?.removeEventListener(
        "devicechange",
        refreshCameras,
      );
    };
  }, [refreshCameras, stop]);

  useEffect(() => {
    const video = videoRef.current;
    const stream = streamRef.current;
    if (!cameraActive || !video || !stream) return;
    video.srcObject = stream;
    void video.play().catch((playError: unknown) => {
      if (!mountedRef.current || streamRef.current !== stream) return;
      stop();
      setStatus("error");
      setError(mediaErrorMessage(playError));
    });
  }, [cameraActive, stop]);

  return {
    videoRef,
    status,
    cameraActive,
    error,
    devices,
    selectedCameraId,
    setSelectedCameraId,
    prediction,
    fps,
    alerts,
    alertsError,
    soundEnabled,
    toggleSound,
    start,
    stop,
  };
}
