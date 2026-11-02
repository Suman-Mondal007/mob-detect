import {
  AlertTriangle,
  CheckCircle2,
  Clock3,
  Download,
  FileVideo2,
  Film,
  HardDriveUpload,
  RefreshCw,
  ShieldCheck,
  Upload,
  X,
} from "lucide-react";
import {
  lazy,
  Suspense,
  useEffect,
  useRef,
  useState,
} from "react";
import type {
  VideoJob,
  VideoResult,
} from "../api/client";
import { getVideoJob, submitVideo } from "../api/client";
import { Badge } from "../components/Badge";
import { Button } from "../components/Button";
import { Card } from "../components/Card";
import { PageIntro } from "../components/PageIntro";
import { Toast } from "../components/Toast";

const VideoTimelineChart = lazy(() =>
  import("../components/VideoTimelineChart").then((module) => ({
    default: module.VideoTimelineChart,
  })),
);

type UploadStatus = "idle" | "uploading" | "processing" | "completed" | "failed";

const ACCEPTED_EXTENSIONS = new Set([".mp4", ".avi", ".mov"]);
const ACCEPTED_MIME_TYPES = new Set([
  "video/mp4",
  "video/x-msvideo",
  "video/avi",
  "video/quicktime",
]);
const configuredMaxUploadMb = Number.parseInt(
  import.meta.env.VITE_MAX_VIDEO_UPLOAD_SIZE_MB ?? "500",
  10,
);
const MAX_UPLOAD_SIZE_MB =
  Number.isFinite(configuredMaxUploadMb) && configuredMaxUploadMb > 0
    ? configuredMaxUploadMb
    : 500;
const MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024;
const POLL_INTERVAL_MS = 1000;

function getValidationError(file: File): string | null {
  const dotIndex = file.name.lastIndexOf(".");
  const extension = dotIndex >= 0 ? file.name.slice(dotIndex).toLowerCase() : "";
  const mimeType = file.type.toLowerCase();
  if (!ACCEPTED_EXTENSIONS.has(extension)) {
    return "Choose an MP4, AVI, or MOV video file.";
  }
  if (
    mimeType &&
    mimeType !== "application/octet-stream" &&
    !ACCEPTED_MIME_TYPES.has(mimeType)
  ) {
    return "The selected file does not appear to be a supported video type.";
  }
  if (file.size === 0) {
    return "The selected video is empty.";
  }
  if (file.size > MAX_UPLOAD_SIZE_BYTES) {
    return `The video exceeds the ${MAX_UPLOAD_SIZE_MB} MB upload limit.`;
  }
  return null;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatTime(seconds: number): string {
  const safeSeconds = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(safeSeconds / 3600);
  const minutes = Math.floor((safeSeconds % 3600) / 60);
  const remainingSeconds = safeSeconds % 60;
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(remainingSeconds).padStart(2, "0")}`
    : `${minutes}:${String(remainingSeconds).padStart(2, "0")}`;
}

function csvCell(value: string | number): string {
  const rawText = String(value);
  const text =
    typeof value === "string" && /^[\t\r ]*[=+\-@]/.test(rawText)
      ? `'${rawText}`
      : rawText;
  return `"${text.replace(/"/g, '""')}"`;
}

function createReportCsv(fileName: string, jobId: string, result: VideoResult) {
  const rows: Array<Array<string | number>> = [
    ["record_type", "video", "job_id", "verdict", "max_violence_probability", "sampled_frames", "start_seconds", "end_seconds", "time_seconds", "violence_probability"],
    ["summary", fileName, jobId, result.verdict, result.max_violence_probability, result.sampled_frames, "", "", "", ""],
    ...result.timeline.map((point) => [
      "timeline",
      fileName,
      jobId,
      "",
      "",
      "",
      "",
      "",
      point.time_seconds,
      point.violence_probability,
    ]),
    ...result.alert_ranges.map((range) => [
      "alert_range",
      fileName,
      jobId,
      "",
      "",
      "",
      range.start_seconds,
      range.end_seconds,
      "",
      "",
    ]),
    ...result.key_frames.map((frame) => [
      "key_frame",
      fileName,
      jobId,
      "",
      "",
      "",
      "",
      "",
      frame.time_seconds,
      frame.violence_probability,
    ]),
  ];
  return rows.map((row) => row.map(csvCell).join(",")).join("\r\n");
}

function downloadReport(
  format: "json" | "csv",
  fileName: string,
  jobId: string,
  result: VideoResult,
) {
  const report =
    format === "json"
      ? JSON.stringify({ job_id: jobId, file_name: fileName, result }, null, 2)
      : createReportCsv(fileName, jobId, result);
  const blob = new Blob([report], {
    type: format === "json" ? "application/json" : "text/csv;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `mob-alert-report-${jobId}.${format}`;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function waitForNextPoll(signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const cleanup = () => signal.removeEventListener("abort", onAbort);
    const timer = window.setTimeout(() => {
      cleanup();
      resolve();
    }, POLL_INTERVAL_MS);
    const onAbort = () => {
      window.clearTimeout(timer);
      cleanup();
      reject(new DOMException("Polling was stopped.", "AbortError"));
    };
    signal.addEventListener(
      "abort",
      onAbort,
      { once: true },
    );
    if (signal.aborted) onAbort();
  });
}

function getJobProgress(job: VideoJob): number {
  return Number.isFinite(job.progress)
    ? Math.min(100, Math.max(0, job.progress))
    : 0;
}

export function UploadPage() {
  const inputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const activeRequestRef = useRef<AbortController | null>(null);
  const objectUrlRef = useRef<string | null>(null);
  const mountedRef = useRef(false);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [status, setStatus] = useState<UploadStatus>("idle");
  const [progress, setProgress] = useState(0);
  const [jobId, setJobId] = useState<string | null>(null);
  const [result, setResult] = useState<VideoResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [previewError, setPreviewError] = useState(false);

  const busy = status === "uploading" || status === "processing";

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      activeRequestRef.current?.abort();
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    };
  }, []);

  function selectFile(selectedFile: File | undefined) {
    if (!selectedFile || busy) return;
    const validationError = getValidationError(selectedFile);
    if (validationError) {
      setError(validationError);
      return;
    }

    activeRequestRef.current?.abort();
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    const objectUrl = URL.createObjectURL(selectedFile);
    objectUrlRef.current = objectUrl;
    setFile(selectedFile);
    setPreviewUrl(objectUrl);
    setPreviewError(false);
    setStatus("idle");
    setProgress(0);
    setJobId(null);
    setResult(null);
    setError(null);
  }

  function clearSelection() {
    if (busy) return;
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    objectUrlRef.current = null;
    setFile(null);
    setPreviewUrl(null);
    setStatus("idle");
    setProgress(0);
    setJobId(null);
    setResult(null);
    setError(null);
    setPreviewError(false);
    if (inputRef.current) inputRef.current.value = "";
  }

  async function pollJob(id: string, controller: AbortController) {
    while (!controller.signal.aborted) {
      const job = await getVideoJob(id, controller.signal);
      if (!mountedRef.current) return;
      setProgress(getJobProgress(job));

      if (job.status === "completed") {
        if (!job.result) {
          throw new Error("The video job completed without returning its results.");
        }
        setResult(job.result);
        setStatus("completed");
        activeRequestRef.current = null;
        return;
      }
      if (job.status === "failed") {
        throw new Error(job.error || "Video processing failed.");
      }
      if (job.status !== "queued" && job.status !== "processing") {
        throw new Error(`The video job returned an unknown status: ${job.status}`);
      }
      setStatus("processing");
      await waitForNextPoll(controller.signal);
    }
  }

  async function startAnalysis() {
    if (!file || busy) return;
    const controller = new AbortController();
    activeRequestRef.current = controller;
    setError(null);
    setStatus("uploading");
    setProgress(0);
    setResult(null);
    setJobId(null);

    try {
      const submittedJob = await submitVideo(file, controller.signal);
      if (!mountedRef.current || controller.signal.aborted) return;
      setJobId(submittedJob.job_id);
      setProgress(getJobProgress(submittedJob));
      setStatus("processing");
      await pollJob(submittedJob.job_id, controller);
    } catch (requestError) {
      if (
        controller.signal.aborted ||
        (requestError instanceof DOMException &&
          requestError.name === "AbortError")
      ) {
        return;
      }
      if (mountedRef.current) {
        setStatus("failed");
        setError(
          requestError instanceof Error
            ? requestError.message
            : "Could not process this video.",
        );
        activeRequestRef.current = null;
      }
    }
  }

  function seekTo(seconds: number) {
    const video = videoRef.current;
    if (!video || !Number.isFinite(seconds)) return;
    video.currentTime = Math.max(0, seconds);
    void video.play().catch(() => {
      video.focus();
    });
  }

  const hasAlert = result?.verdict === "VIOLENCE_ALERT";

  return (
    <div className="page-container upload-page">
      <PageIntro
        eyebrow="Video analysis"
        title="Review a recording."
        description="Submit a video to screen sampled frames, inspect the probability timeline, and review alert intervals and key frames."
        action={
          <Badge tone={busy ? "brand" : result ? (hasAlert ? "danger" : "success") : "neutral"}>
            {busy ? "Analysis in progress" : result ? "Analysis complete" : "MP4 · AVI · MOV"}
          </Badge>
        }
      />

      {error && (
        <Toast
          tone="error"
          title={status === "failed" ? "Video analysis failed" : "Could not use this video"}
          description={error}
          onDismiss={() => setError(null)}
        />
      )}

      {!file ? (
        <Card
          className={`upload-dropzone ${dragActive ? "is-drag-active" : ""}`}
          onDragEnter={(event) => {
            event.preventDefault();
            if (!busy) setDragActive(true);
          }}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={(event) => {
            if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
              setDragActive(false);
            }
          }}
          onDrop={(event) => {
            event.preventDefault();
            setDragActive(false);
            selectFile(event.dataTransfer.files[0]);
          }}
        >
          <input
            ref={inputRef}
            className="upload-file-input"
            type="file"
            accept=".mp4,.avi,.mov,video/mp4,video/x-msvideo,video/quicktime"
            onChange={(event) => selectFile(event.currentTarget.files?.[0])}
            aria-label="Choose a video to analyze"
            aria-hidden="true"
            tabIndex={-1}
          />
          <span className="upload-dropzone-icon">
            <HardDriveUpload size={27} aria-hidden="true" />
          </span>
          <h2 className="upload-dropzone-title">Drop a video to get started</h2>
          <p className="upload-dropzone-copy">
            Choose a video from your device or drag it into this area.
          </p>
          <Button
            variant="primary"
            size="md"
            onClick={() => inputRef.current?.click()}
            icon={<Upload size={16} aria-hidden="true" />}
          >
            Browse videos
          </Button>
          <p className="upload-dropzone-limits">
            MP4, AVI, or MOV · Up to {MAX_UPLOAD_SIZE_MB} MB
          </p>
        </Card>
      ) : (
        <div className="upload-workspace">
          <Card className="upload-file-card">
            <div className="upload-file-heading">
              <span className="upload-file-icon">
                <FileVideo2 size={19} aria-hidden="true" />
              </span>
              <div className="upload-file-info">
                <p className="upload-file-name" title={file.name}>{file.name}</p>
                <p className="upload-file-meta">{formatBytes(file.size)} · {file.name.split(".").pop()?.toUpperCase()}</p>
              </div>
              {!busy && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={clearSelection}
                  aria-label="Remove selected video"
                  icon={<X size={16} aria-hidden="true" />}
                />
              )}
            </div>

            {previewUrl && (
              <div className="upload-video-preview">
                <video
                  ref={videoRef}
                  src={previewUrl}
                  controls
                  playsInline
                  preload="metadata"
                  onError={() => setPreviewError(true)}
                  aria-label={`Preview of ${file.name}`}
                />
                {previewError && (
                  <p className="upload-preview-note">
                    This browser cannot preview the video format. The backend may still be able to analyze it.
                  </p>
                )}
              </div>
            )}

            <div className="upload-file-actions">
              <Button
                variant="primary"
                size="lg"
                onClick={() => void startAnalysis()}
                disabled={busy}
                icon={
                  busy ? (
                    <RefreshCw className="upload-spin" size={16} aria-hidden="true" />
                  ) : (
                    <Film size={16} aria-hidden="true" />
                  )
                }
              >
                {status === "uploading"
                  ? "Uploading video…"
                  : status === "processing"
                    ? "Analyzing video…"
                    : status === "failed" && jobId
                      ? "Try analysis again"
                      : "Analyze video"}
              </Button>
              {status === "failed" && jobId && (
                <span className="upload-job-id">Job {jobId.slice(0, 8)}</span>
              )}
            </div>
          </Card>

          {(status === "uploading" || status === "processing") && (
            <Card className="upload-progress-card" role="status" aria-live="polite">
              <div className="upload-progress-heading">
                <span className="upload-progress-icon">
                  {status === "uploading" ? (
                    <Upload size={16} aria-hidden="true" />
                  ) : (
                    <RefreshCw className="upload-spin" size={16} aria-hidden="true" />
                  )}
                </span>
                <div>
                  <p className="upload-progress-title">
                    {status === "uploading" ? "Uploading video" : "Analyzing sampled frames"}
                  </p>
                  <p className="upload-progress-copy">
                    {status === "uploading"
                      ? "Sending the video to the analysis service…"
                      : jobId
                        ? `Job ${jobId.slice(0, 8)} · status updates every second`
                        : "Preparing video-analysis job…"}
                  </p>
                </div>
                <span className="upload-progress-percent">
                  {status === "uploading" ? "…" : `${Math.round(progress)}%`}
                </span>
              </div>
              <div
                className={`upload-progress-track ${status === "uploading" ? "is-indeterminate" : ""}`}
                role="progressbar"
                aria-label={status === "uploading" ? "Uploading video" : "Video analysis progress"}
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={status === "uploading" ? undefined : Math.round(progress)}
              >
                <span style={status === "uploading" ? undefined : { width: `${progress}%` }} />
              </div>
            </Card>
          )}

          {result && jobId && (
            <div className="upload-results">
              <Card className={`upload-verdict-card ${hasAlert ? "has-alert" : "is-clear"}`}>
                <span className="upload-verdict-icon">
                  {hasAlert ? (
                    <AlertTriangle size={23} aria-hidden="true" />
                  ) : (
                    <ShieldCheck size={23} aria-hidden="true" />
                  )}
                </span>
                <div className="upload-verdict-copy">
                  <p className="upload-verdict-eyebrow">Overall screening verdict</p>
                  <h2>{hasAlert ? "Potential violence detected" : "No violence alert detected"}</h2>
                  <p>
                    {hasAlert
                      ? "One or more sampled time ranges crossed the configured alert threshold. Review the footage directly."
                      : "No sampled interval crossed the configured alert threshold. This does not guarantee the video contains no violence."}
                  </p>
                </div>
                <div className="upload-verdict-stat">
                  <strong>{Math.round(result.max_violence_probability * 100)}%</strong>
                  <span>maximum probability</span>
                </div>
              </Card>

              <div className="upload-report-actions">
                <span className="upload-results-meta">
                  <CheckCircle2 size={15} aria-hidden="true" />
                  {result.sampled_frames} sampled frames · Job {jobId.slice(0, 8)}
                </span>
                <div className="upload-report-buttons">
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => downloadReport("json", file.name, jobId, result)}
                    icon={<Download size={14} aria-hidden="true" />}
                  >
                    Download JSON
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => downloadReport("csv", file.name, jobId, result)}
                    icon={<Download size={14} aria-hidden="true" />}
                  >
                    Download CSV
                  </Button>
                </div>
              </div>

              <Card className="upload-chart-card">
                <div className="upload-section-heading">
                  <div>
                    <h2>Violence probability timeline</h2>
                    <p>Click a point to seek the video to that time.</p>
                  </div>
                  <Badge tone="neutral">{result.timeline.length} points</Badge>
                </div>
                {result.timeline.length > 0 ? (
                  <Suspense fallback={<div className="upload-chart-loading">Loading timeline chart…</div>}>
                    <VideoTimelineChart result={result} onSeek={seekTo} />
                  </Suspense>
                ) : (
                  <p className="upload-no-data">The job returned no timeline points.</p>
                )}
              </Card>

              <div className="upload-results-grid">
                <Card className="upload-ranges-card">
                  <div className="upload-section-heading">
                    <div>
                      <h2>Alert time ranges</h2>
                      <p>Intervals where the alert was active.</p>
                    </div>
                    <Badge tone={result.alert_ranges.length ? "danger" : "success"}>
                      {result.alert_ranges.length}
                    </Badge>
                  </div>
                  {result.alert_ranges.length > 0 ? (
                    <ul className="upload-range-list">
                      {result.alert_ranges.map((range, index) => (
                        <li key={`${range.start_seconds}-${index}`}>
                          <span className="upload-range-number">{String(index + 1).padStart(2, "0")}</span>
                          <span className="upload-range-time">
                            {formatTime(range.start_seconds)} – {formatTime(range.end_seconds)}
                          </span>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => seekTo(range.start_seconds)}
                            aria-label={`Play alert range starting at ${formatTime(range.start_seconds)}`}
                          >
                            Play
                          </Button>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <div className="upload-no-alerts">
                      <CheckCircle2 size={18} aria-hidden="true" />
                      <span>No alert ranges were returned for this video.</span>
                    </div>
                  )}
                </Card>

                <Card className="upload-keyframes-card">
                  <div className="upload-section-heading">
                    <div>
                      <h2>Key frames</h2>
                      <p>Highest-probability sampled moments.</p>
                    </div>
                    <Badge tone="neutral">{result.key_frames.length} / 5</Badge>
                  </div>
                  {result.key_frames.length > 0 ? (
                    <div className="upload-keyframes-grid">
                      {result.key_frames.map((frame, index) => (
                        <button
                          className="upload-keyframe"
                          key={`${frame.time_seconds}-${index}`}
                          onClick={() => seekTo(frame.time_seconds)}
                          type="button"
                          aria-label={`Seek to key frame at ${formatTime(frame.time_seconds)}, probability ${Math.round(frame.violence_probability * 100)} percent`}
                        >
                          <img
                            src={`data:image/jpeg;base64,${frame.image_base64}`}
                            alt={`Key frame at ${formatTime(frame.time_seconds)}`}
                            loading="lazy"
                          />
                          <span className="upload-keyframe-meta">
                            <span>{formatTime(frame.time_seconds)}</span>
                            <strong>{Math.round(frame.violence_probability * 100)}%</strong>
                          </span>
                        </button>
                      ))}
                    </div>
                  ) : (
                    <div className="upload-no-alerts">
                      <Clock3 size={18} aria-hidden="true" />
                      <span>No key frames were returned for this video.</span>
                    </div>
                  )}
                </Card>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
