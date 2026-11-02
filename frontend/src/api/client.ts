export function startCamera(cameraId: string): Promise<ManagedCamera> {
  return request<ManagedCamera>(`/cameras/${encodeURIComponent(cameraId)}/start`, {
    method: "POST",
  });
}

export function stopCamera(cameraId: string): Promise<ManagedCamera> {
  return request<ManagedCamera>(`/cameras/${encodeURIComponent(cameraId)}/stop`, {
    method: "POST",
  });
}

export async function getSystemHealth(): Promise<{
  uptime_seconds: number;
  gpu_memory: { allocated_bytes: number; reserved_bytes: number } | null;
  cameras: Array<{
    camera_id: string;
    online: boolean;
    input_fps: number;
    inference_fps: number;
    inference_ms: number | null;
    state: string;
    error: string | null;
  }>;
}> {
  return request("/health/system");
}

export async function login(username: string, password: string): Promise<{ access_token: string; role: string; username: string }> {
  const response = await fetch(`${API_URL}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const body: unknown = await response.json();
      if (typeof body === "object" && body !== null && "detail" in body && typeof body.detail === "string") {
        detail = body.detail;
      }
    } catch {
      // Keep status text.
    }
    throw new Error(detail);
  }
  return (await response.json()) as { access_token: string; role: string; username: string };
}

const configuredApiUrl = import.meta.env.VITE_API_URL?.trim();
export const API_URL = (configuredApiUrl || "http://127.0.0.1:8000").replace(
  /\/+$/,
  "",
);

export interface HealthResponse {
  model_loaded: boolean;
  device: string;
  version: string;
}

export interface ImagePrediction {
  predicted_class: string;
  confidence: number;
  violence_probability: number;
}

export interface VideoJob {
  job_id: string;
  status: string;
  progress: number;
  result?: VideoResult | null;
  error?: string | null;
}

export interface VideoTimelinePoint {
  time_seconds: number;
  violence_probability: number;
}

export interface VideoAlertRange {
  start_seconds: number;
  end_seconds: number;
}

export interface VideoKeyFrame {
  time_seconds: number;
  violence_probability: number;
  image_base64: string;
}

export interface VideoResult {
  verdict: string;
  max_violence_probability: number;
  sampled_frames: number;
  timeline: VideoTimelinePoint[];
  alert_ranges: VideoAlertRange[];
  key_frames: VideoKeyFrame[];
}

export interface LiveAlert {
  id: number;
  timestamp: string;
  source: "live" | "upload";
  source_name: string;
  label: string;
  confidence: number;
  smoothed_confidence: number;
  threshold: number;
  snapshot_url: string | null;
  video_time_seconds: number | null;
}

export interface AlertStats {
  total_alerts: number;
  last_24h: number;
  average_confidence: number;
}

export interface AlertListResponse {
  alerts: LiveAlert[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
  stats: AlertStats;
}

export interface AlertQuery {
  page?: number;
  pageSize?: number;
  source?: "live" | "upload";
  search?: string;
  signal?: AbortSignal;
}

export interface CameraStatus {
  camera_id: string;
  source: string;
  source_type: "file" | "url" | "webcam" | "rtsp";
  online: boolean;
  last_frame_at: string | null;
  last_inference_at: string | null;
  inference_ms: number | null;
  predicted_class: string | null;
  confidence: number | null;
  smoothed_confidence: number | null;
  alert: boolean;
  state: "NORMAL" | "ALERT" | "ACKNOWLEDGED" | "COOLDOWN";
  input_fps: number;
  inference_fps: number;
  error: string | null;
}

export interface ManagedCamera extends CameraStatus {
  id: string;
  name: string;
  location: string;
  source_uri: string;
  enabled: boolean;
  created_at: string;
}

export interface CameraListResponse {
  cameras: ManagedCamera[];
}
export interface CameraPreviewResponse {
  ok: boolean;
  error: string | null;
  preview_frame_base64: string | null;
  preview_mime_type: "image/jpeg" | null;
}

export interface IncidentQuery {
  page?: number;
  pageSize?: number;
  cameraId?: string;
  status?: "new" | "acknowledged" | "false-positive";
  startTime?: string;
  endTime?: string;
  search?: string;
  signal?: AbortSignal;
}


export interface Incident {
  id: number;
  camera_id: string;
  start_time: string;
  end_time: string | null;
  peak_confidence: number;
  snapshot_url: string | null;
  clip_url: string | null;
  status: "new" | "acknowledged" | "false-positive";
}

export interface IncidentListResponse {
  incidents: Incident[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface CameraStatusEvent {
  type: "camera_status";
  timestamp: string;
  cameras: CameraStatus[];
}

export interface CameraAlertEvent {
  type: "alert";
  timestamp: string;
  camera_id: string;
  confidence: number | null;
  smoothed_confidence: number | null;
  state: "ALERT";
}

export interface CameraHeartbeatEvent {
  type: "heartbeat";
  timestamp: number;
}

export type CameraEvent =
  | CameraStatusEvent
  | CameraAlertEvent
  | CameraHeartbeatEvent;

export function getLiveWebSocketUrl(): string {
  const url = new URL(API_URL);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = `${url.pathname.replace(/\/+$/, "")}/ws/live`;
  const token = getAuthToken();
  if (token) url.searchParams.set("token", token);
  return url.toString();
}

export function getCameraEventsWebSocketUrl(): string {
  const url = new URL(API_URL);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = `${url.pathname.replace(/\/+$/, "")}/ws/events`;
  const token = getAuthToken();
  if (token) url.searchParams.set("token", token);
  return url.toString();
}

export function getCameraStreamUrl(
  cameraId: string,
  quality: "grid" | "full" = "grid",
): string {
  const url = new URL(`${API_URL}/stream/${encodeURIComponent(cameraId)}`);
  url.searchParams.set("quality", quality);
  const token = getAuthToken();
  if (token) url.searchParams.set("token", token);
  return url.toString();
}

export function getAuthToken(): string | null {
  try {
    return window.localStorage.getItem("mob-alert-auth-token");
  } catch {
    return null;
  }
}

export function setAuthToken(token: string | null): void {
  try {
    if (token) window.localStorage.setItem("mob-alert-auth-token", token);
    else window.localStorage.removeItem("mob-alert-auth-token");
  } catch {
    // Ignore storage failures.
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const token = getAuthToken();
  const headers = new Headers(init?.headers);
  if (token) headers.set("Authorization", `Bearer ${token}`);
  let response: Response;
  try {
    response = await fetch(`${API_URL}${path}`, { ...init, headers });
  } catch (error) {
    if (init?.signal?.aborted) throw error;
    const message = error instanceof Error ? error.message : "Network request failed.";
    throw new Error(`Could not connect to the Mob-Alert API: ${message}`);
  }

  if (response.status === 401) {
    setAuthToken(null);
    if (typeof window !== "undefined") window.location.replace("/login");
  }

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}.`;
    try {
      const body: unknown = await response.json();
      if (
        typeof body === "object" &&
        body !== null &&
        "detail" in body &&
        typeof body.detail === "string"
      ) {
        detail = body.detail;
      }
    } catch {
      // Retain the explicit HTTP status when the server does not return JSON.
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export function getCameras(signal?: AbortSignal): Promise<CameraListResponse> {
  return request<CameraListResponse>("/cameras", { signal });
}

export function getIncidents(query: IncidentQuery = {}): Promise<IncidentListResponse> {
  const searchParams = new URLSearchParams();
  if (query.page !== undefined) searchParams.set("page", String(query.page));
  if (query.pageSize !== undefined) {
    searchParams.set("page_size", String(query.pageSize));
  }
  if (query.cameraId) searchParams.set("camera_id", query.cameraId);
  if (query.status) searchParams.set("status", query.status);
  if (query.startTime) searchParams.set("start_time", query.startTime);
  if (query.endTime) searchParams.set("end_time", query.endTime);
  if (query.search) searchParams.set("search", query.search);
  const suffix = searchParams.size ? `?${searchParams.toString()}` : "";
  return request<IncidentListResponse>(`/incidents${suffix}`, {
    signal: query.signal,
  });
}

export function acknowledgeIncident(incidentId: number): Promise<Incident> {
  return request<Incident>(
    `/incidents/${encodeURIComponent(incidentId)}/acknowledge`,
    { method: "POST" },
  );
}

export function markIncidentFalsePositive(incidentId: number): Promise<Incident> {
  return request<Incident>(
    `/incidents/${encodeURIComponent(incidentId)}/false-positive`,
    { method: "POST" },
  );
}

export function acknowledgeCameraAlert(cameraId: string): Promise<CameraStatus> {
  return request<CameraStatus>(
    `/cameras/${encodeURIComponent(cameraId)}/acknowledge`,
    { method: "POST" },
  );
}

export function getIncidentSnapshotUrl(snapshotPath: string): string {
  return new URL(snapshotPath, `${API_URL}/`).toString();
}

export function predictImage(file: File): Promise<ImagePrediction> {
  const body = new FormData();
  body.append("file", file);
  return request<ImagePrediction>("/predict-image", { method: "POST", body });
}

export function submitVideo(
  file: File,
  signal?: AbortSignal,
): Promise<VideoJob> {
  const body = new FormData();
  body.append("file", file);
  return request<VideoJob>("/predict-video", {
    method: "POST",
    body,
    signal,
  });
}

export function getVideoJob(
  jobId: string,
  signal?: AbortSignal,
): Promise<VideoJob> {
  return request<VideoJob>(`/jobs/${encodeURIComponent(jobId)}`, { signal });
}

export function getAlerts(query: AlertQuery = {}): Promise<AlertListResponse> {
  const searchParams = new URLSearchParams();
  if (query.page !== undefined) searchParams.set("page", String(query.page));
  if (query.pageSize !== undefined) {
    searchParams.set("page_size", String(query.pageSize));
  }
  if (query.source) searchParams.set("source", query.source);
  if (query.search) searchParams.set("search", query.search);
  const suffix = searchParams.size ? `?${searchParams.toString()}` : "";
  return request<AlertListResponse>(`/alerts${suffix}`, {
    signal: query.signal,
  });
}

export function getAlertSnapshotUrl(snapshotPath: string): string {
  return `${API_URL}${snapshotPath}`;
}

export function getAlertsCsvUrl(
  filters: Pick<AlertQuery, "source" | "search"> = {},
): string {
  const searchParams = new URLSearchParams();
  if (filters.source) searchParams.set("source", filters.source);
  if (filters.search) searchParams.set("search", filters.search);
  const suffix = searchParams.size ? `?${searchParams.toString()}` : "";
  return `${API_URL}/alerts.csv${suffix}`;
}

export function createCamera(payload: {
  id?: string;
  name: string;
  location?: string;
  source_type: "rtsp" | "file" | "webcam";
  source_uri: string;
  enabled?: boolean;
  inference_fps?: number;
}): Promise<ManagedCamera> {
  return request<ManagedCamera>("/cameras", {
    method: "POST",
    body: JSON.stringify(payload),
    headers: { "Content-Type": "application/json" },
  });
}

export function updateCamera(
  cameraId: string,
  changes: {
    name?: string;
    location?: string;
    source_type?: "rtsp" | "file" | "webcam";
    source_uri?: string;
    enabled?: boolean;
    inference_fps?: number;
  },
): Promise<ManagedCamera> {
  return request<ManagedCamera>(`/cameras/${encodeURIComponent(cameraId)}`, {
    method: "PUT",
    body: JSON.stringify(changes),
    headers: { "Content-Type": "application/json" },
  });
}

export async function deleteCamera(cameraId: string): Promise<void> {
  const token = getAuthToken();
  const headers: HeadersInit = token ? { Authorization: `Bearer ${token}` } : {};
  const response = await fetch(
    `${API_URL}/cameras/${encodeURIComponent(cameraId)}`,
    { method: "DELETE", headers },
  );
  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const body: unknown = await response.json();
      if (
        typeof body === "object" &&
        body !== null &&
        "detail" in body &&
        typeof body.detail === "string"
      ) {
        detail = body.detail;
      }
    } catch {
      // Keep status text.
    }
    throw new Error(detail);
  }
}

export function testCamera(cameraId: string): Promise<CameraPreviewResponse> {
  return request<CameraPreviewResponse>(
    `/cameras/${encodeURIComponent(cameraId)}/test`,
    { method: "POST" },
  );
}

export function getIncidentsCsvUrl(filters: {
  cameraId?: string;
  status?: "new" | "acknowledged" | "false-positive";
  startTime?: string;
  endTime?: string;
  search?: string;
} = {}): string {
  const searchParams = new URLSearchParams();
  if (filters.cameraId) searchParams.set("camera_id", filters.cameraId);
  if (filters.status) searchParams.set("status", filters.status);
  if (filters.startTime) searchParams.set("start_time", filters.startTime);
  if (filters.endTime) searchParams.set("end_time", filters.endTime);
  if (filters.search) searchParams.set("search", filters.search);
  const suffix = searchParams.size ? `?${searchParams.toString()}` : "";
  return `${API_URL}/incidents/export.csv${suffix}`;
}
