import {
  AlertTriangle,
  Activity,
  BellRing,
  Camera,
  CheckCircle2,
  CircleDot,
  Clock3,
  Radio,
  ShieldCheck,
  Volume2,
  VolumeX,
} from "lucide-react";
import { useEffect, useRef } from "react";
import { Badge } from "../components/Badge";
import { Button } from "../components/Button";
import { Card } from "../components/Card";
import { PageIntro } from "../components/PageIntro";
import { Spinner } from "../components/Spinner";
import { Toast } from "../components/Toast";
import { useLiveDetection } from "../hooks/useLiveDetection";

function percentage(value: number | undefined): string {
  return typeof value === "number" && Number.isFinite(value)
    ? `${Math.round(Math.min(1, Math.max(0, value)) * 100)}%`
    : "—";
}

function formatTime(timestamp: string): string {
  const date = new Date(timestamp);
  return Number.isNaN(date.getTime())
    ? timestamp
    : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function LivePage() {
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const {
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
  } = useLiveDetection();

  const active = status === "requesting" || status === "connecting" || status === "live" || status === "reconnecting";
  const statusDetails: Record<
    typeof status,
    { label: string; tone: "neutral" | "brand" | "warning" | "danger"; dot: string }
  > = {
    disconnected: { label: "Standby", tone: "neutral", dot: "idle" },
    requesting: { label: "Requesting camera", tone: "warning", dot: "working" },
    connecting: { label: "Connecting", tone: "warning", dot: "working" },
    live: { label: "Live screening", tone: "brand", dot: "live" },
    reconnecting: { label: "Reconnecting", tone: "warning", dot: "working" },
    error: { label: "Needs attention", tone: "danger", dot: "error" },
  };
  const displayStatus = statusDetails[status];

  useEffect(() => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    const draw = () => {
      const bounds = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const backingWidth = Math.max(1, Math.round(bounds.width * ratio));
      const backingHeight = Math.max(1, Math.round(bounds.height * ratio));
      if (canvas.width !== backingWidth) canvas.width = backingWidth;
      if (canvas.height !== backingHeight) canvas.height = backingHeight;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, bounds.width, bounds.height);
      if (!cameraActive) return;

      const inset = 18;
      const corner = 23;
      context.strokeStyle = prediction?.alert
        ? "rgba(255, 91, 87, 0.94)"
        : "rgba(108, 218, 196, 0.86)";
      context.lineWidth = 2;
      context.beginPath();
      context.moveTo(inset, inset + corner);
      context.lineTo(inset, inset);
      context.lineTo(inset + corner, inset);
      context.moveTo(bounds.width - inset - corner, inset);
      context.lineTo(bounds.width - inset, inset);
      context.lineTo(bounds.width - inset, inset + corner);
      context.moveTo(inset, bounds.height - inset - corner);
      context.lineTo(inset, bounds.height - inset);
      context.lineTo(inset + corner, bounds.height - inset);
      context.moveTo(bounds.width - inset - corner, bounds.height - inset);
      context.lineTo(bounds.width - inset, bounds.height - inset);
      context.lineTo(bounds.width - inset, bounds.height - inset - corner);
      context.stroke();

      if (prediction) {
        const label = `${prediction.label}  ·  ${percentage(prediction.confidence)} confidence`;
        context.font = "600 12px system-ui, sans-serif";
        const width = Math.min(bounds.width - 36, context.measureText(label).width + 24);
        context.fillStyle = "rgba(13, 19, 25, 0.78)";
        context.beginPath();
        context.roundRect(18, 18, width, 34, 8);
        context.fill();
        context.fillStyle = "#ffffff";
        context.fillText(label, 30, 40, width - 18);
      }
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [cameraActive, prediction]);

  return (
    <div className="page-container live-page">
      <PageIntro
        eyebrow="Live monitoring"
        title="Stay aware, in real time."
        description="Screen a live camera feed for possible violence. The smooth preview stays on your device while small frames are sent to the analysis service."
        action={
          <Badge tone={displayStatus.tone}>
            <CircleDot size={13} aria-hidden="true" />
            {displayStatus.label}
          </Badge>
        }
      />

      {prediction?.alert && (
        <div className="live-alert-banner" role="alert" aria-live="assertive">
          <span className="live-alert-banner-icon">
            <AlertTriangle size={22} aria-hidden="true" />
          </span>
          <div>
            <p className="live-alert-title">Potential violence detected</p>
            <p className="live-alert-copy">
              Screening result only — verify the situation directly.
            </p>
          </div>
          <span className="live-alert-confidence">
            {percentage(prediction.smoothed_confidence)}
          </span>
        </div>
      )}

      <div className="live-dashboard">
        <div className="live-main-column">
          <Card className="live-preview-card">
            <div className="preview-toolbar">
              <div className="flex items-center gap-3">
                <span className="preview-icon">
                  <Camera size={19} aria-hidden="true" />
                </span>
                <div>
                  <h2 className="text-sm font-semibold text-ink">Camera preview</h2>
                  <p className="mt-0.5 text-xs text-muted">
                    {cameraActive ? "Local preview · frames sent for analysis" : "Camera remains off until you start"}
                  </p>
                </div>
              </div>
              <Badge tone={displayStatus.tone}>
                <span className={`status-dot status-dot-${displayStatus.dot}`} />
                {displayStatus.label}
              </Badge>
            </div>

            <div className={`preview-stage live-preview-stage ${cameraActive ? "is-camera-active" : ""}`}>
              <div className="preview-grid" aria-hidden="true" />
              {cameraActive ? (
                <>
                  <video
                    ref={videoRef}
                    className="live-video"
                    autoPlay
                    muted
                    playsInline
                    aria-label="Local webcam preview"
                  />
                  <canvas ref={overlayRef} className="live-canvas-overlay" aria-hidden="true" />
                  {prediction?.alert && (
                    <span className="live-video-alert">
                      <AlertTriangle size={14} aria-hidden="true" />
                      ALERT
                    </span>
                  )}
                  <span className="live-video-indicator">
                    <span className={`status-dot status-dot-${displayStatus.dot}`} />
                    {status === "live" ? "LIVE" : displayStatus.label.toUpperCase()}
                  </span>
                </>
              ) : (
                <div className="preview-empty">
                  <span className="preview-empty-icon">
                    {status === "requesting" ? (
                      <Spinner label="Waiting for camera permission" />
                    ) : (
                      <Camera size={27} strokeWidth={1.7} aria-hidden="true" />
                    )}
                  </span>
                  <p className="text-sm font-medium text-ink">
                    {status === "requesting" ? "Waiting for camera access" : "Your preview will appear here"}
                  </p>
                  <p className="mt-1 text-xs text-muted">
                    {status === "requesting"
                      ? "Allow camera access in your browser to continue."
                      : "Your camera is only activated after you select Start live screening."}
                  </p>
                </div>
              )}
              <span className="preview-corner preview-corner-tl" aria-hidden="true" />
              <span className="preview-corner preview-corner-tr" aria-hidden="true" />
              <span className="preview-corner preview-corner-bl" aria-hidden="true" />
              <span className="preview-corner preview-corner-br" aria-hidden="true" />
            </div>

            <div className="preview-footer live-preview-footer">
              <span className="flex items-center gap-2 text-xs text-muted">
                <span className={`status-dot status-dot-${displayStatus.dot}`} />
                {cameraActive
                  ? status === "live"
                    ? "Analysis is running"
                    : "Camera preview is running"
                  : "Inference is not running"}
              </span>
              <span className="text-xs text-muted">
                {cameraActive ? "Only downscaled frames are sent" : "No camera connected"}
              </span>
            </div>
          </Card>

          <div className="live-controls">
            <label className="live-camera-field">
              <span className="live-field-label">Camera</span>
              <select
                value={selectedCameraId}
                onChange={(event) => setSelectedCameraId(event.target.value)}
                disabled={cameraActive || status === "requesting" || devices.length === 0}
                aria-label="Select camera"
              >
                {devices.length === 0 ? (
                  <option value="">No camera detected yet</option>
                ) : (
                  devices.map((device, index) => (
                    <option key={device.deviceId} value={device.deviceId}>
                      {device.label || `Camera ${index + 1}`}
                    </option>
                  ))
                )}
              </select>
            </label>
            <div className="live-action-buttons">
              <Button
                variant={active ? "danger" : "primary"}
                size="lg"
                onClick={active ? stop : () => void start()}
                disabled={status === "requesting"}
                icon={
                  active ? (
                    <CircleDot size={16} aria-hidden="true" />
                  ) : (
                    <Radio size={16} aria-hidden="true" />
                  )
                }
              >
                {status === "requesting"
                  ? "Requesting camera…"
                  : active
                    ? "Stop screening"
                    : status === "error"
                      ? "Try again"
                      : "Start live screening"}
              </Button>
              <Button
                variant="secondary"
                size="lg"
                onClick={() => void toggleSound()}
                aria-pressed={soundEnabled}
                aria-label={soundEnabled ? "Turn alert sound off" : "Turn alert sound on"}
                title={soundEnabled ? "Turn alert sound off" : "Turn alert sound on"}
                icon={
                  soundEnabled ? (
                    <Volume2 size={17} aria-hidden="true" />
                  ) : (
                    <VolumeX size={17} aria-hidden="true" />
                  )
                }
              >
                Sound {soundEnabled ? "on" : "off"}
              </Button>
            </div>
          </div>

          {error && (
            <Toast
              tone="error"
              title={status === "error" ? "Could not start live screening" : "Live screening issue"}
              description={error}
            />
          )}

          <div className="live-metrics-grid" aria-label="Live analysis metrics">
            <Card className="live-metric-card">
              <span className="live-metric-icon live-metric-brand">
                <Activity size={17} aria-hidden="true" />
              </span>
              <div>
                <p className="status-label">Detection label</p>
                <p className="status-value">{prediction?.label ?? "Waiting for result"}</p>
              </div>
            </Card>
            <Card className="live-metric-card live-confidence-card">
              <div className="live-metric-heading">
                <span className="status-label">Violence probability</span>
                <span className="live-confidence-value">
                  {percentage(prediction?.smoothed_confidence)}
                </span>
              </div>
              <div
                className="live-confidence-track"
                role="progressbar"
                aria-label="Smoothed violence probability"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={
                  prediction
                    ? Math.round(prediction.smoothed_confidence * 100)
                    : 0
                }
              >
                <span
                  className={`live-confidence-fill ${prediction?.alert ? "is-alert" : ""}`}
                  style={{
                    width: prediction
                      ? `${Math.min(100, Math.max(0, prediction.smoothed_confidence * 100))}%`
                      : "0%",
                  }}
                />
              </div>
              <span className="live-metric-caption">
                Raw confidence {percentage(prediction?.confidence)}
              </span>
            </Card>
            <Card className="live-metric-card">
              <span className="live-metric-icon live-metric-neutral">
                <Clock3 size={17} aria-hidden="true" />
              </span>
              <div>
                <p className="status-label">Inference latency</p>
                <p className="status-value">
                  {prediction ? Math.round(prediction.inference_ms) : "—"}
                  <span className="status-unit"> ms</span>
                </p>
              </div>
            </Card>
            <Card className="live-metric-card">
              <span className="live-metric-icon live-metric-success">
                <Radio size={17} aria-hidden="true" />
              </span>
              <div>
                <p className="status-label">Result rate</p>
                <p className="status-value">{fps.toFixed(1)} <span className="status-unit">FPS</span></p>
              </div>
            </Card>
          </div>
        </div>

        <Card className="live-alerts-card">
          <div className="live-panel-heading">
            <span className="live-panel-icon">
              <BellRing size={17} aria-hidden="true" />
            </span>
            <div>
              <h2 className="live-panel-title">Latest alerts</h2>
              <p className="live-panel-subtitle">Recent alert events from this service</p>
            </div>
            <Badge tone="neutral">{alerts.length}</Badge>
          </div>
          {alertsError && (
            <p className="live-alerts-error" role="status">{alertsError}</p>
          )}
          {alerts.length > 0 ? (
            <ul className="live-alert-list">
              {alerts.map((alert, index) => (
                <li className="live-alert-item" key={`${alert.timestamp}-${index}`}>
                  <span className="live-alert-item-icon">
                    <AlertTriangle size={15} aria-hidden="true" />
                  </span>
                  <span className="live-alert-item-content">
                    <span className="live-alert-item-title">{alert.label}</span>
                    <span className="live-alert-item-time">{formatTime(alert.timestamp)}</span>
                  </span>
                  <span className="live-alert-item-confidence">
                    {percentage(alert.smoothed_confidence)}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <div className="live-alerts-empty">
              <span className="live-alerts-empty-icon">
                {active ? <Activity size={19} aria-hidden="true" /> : <ShieldCheck size={19} aria-hidden="true" />}
              </span>
              <p>{active ? "No alerts detected" : "No recent alerts"}</p>
              <span>
                {active
                  ? "Alert events will appear here if the configured threshold is reached."
                  : "Start live screening to monitor a camera feed."}
              </span>
            </div>
          )}
          <div className="live-panel-footnote">
            <CheckCircle2 size={14} aria-hidden="true" />
            Alerts are screening signals, not confirmed incidents.
          </div>
        </Card>
      </div>
    </div>
  );
}
