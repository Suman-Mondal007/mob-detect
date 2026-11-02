import {
  Camera as CameraIcon,
  CheckCircle2,
  GripVertical,
  Pencil,
  Plus,
  Power,
  Search,
  Trash2,
  X,
} from "lucide-react";
import { useCallback, useEffect, useState, type FormEvent } from "react";
import {
  createCamera,
  deleteCamera,
  getCameras,
  startCamera,
  stopCamera,
  testCamera,
  updateCamera,
  type ManagedCamera,
} from "../api/client";
import { Badge } from "../components/Badge";
import { Button } from "../components/Button";
import { Card } from "../components/Card";
import { Modal } from "../components/Modal";
import { PageIntro } from "../components/PageIntro";
import { Spinner } from "../components/Spinner";
import { Toast } from "../components/Toast";

const CAMERA_PREFS_KEY = "mob-alert-camera-preferences";
const CAMERA_ORDER_KEY = "mob-alert-camera-order";

interface CameraPrefs {
  threshold: number;
  sensitivity: "low" | "balanced" | "high";
}

type CameraPrefsMap = Record<string, CameraPrefs>;

function loadPrefs(): CameraPrefsMap {
  try {
    const stored = window.localStorage.getItem(CAMERA_PREFS_KEY);
    if (!stored) return {};
    const parsed = JSON.parse(stored);
    return typeof parsed === "object" && parsed !== null ? parsed : {};
  } catch {
    return {};
  }
}

function savePrefs(prefs: CameraPrefsMap) {
  try {
    window.localStorage.setItem(CAMERA_PREFS_KEY, JSON.stringify(prefs));
  } catch {
    // Ignore storage failures.
  }
}

function loadOrder(): string[] {
  try {
    const stored = window.localStorage.getItem(CAMERA_ORDER_KEY);
    if (!stored) return [];
    const parsed = JSON.parse(stored);
    return Array.isArray(parsed) ? parsed.filter((item) => typeof item === "string") : [];
  } catch {
    return [];
  }
}

function saveOrder(order: string[]) {
  try {
    window.localStorage.setItem(CAMERA_ORDER_KEY, JSON.stringify(order));
  } catch {
    // Ignore storage failures.
  }
}

function sortByOrder(cameras: ManagedCamera[], order: string[]): ManagedCamera[] {
  const index = new Map(order.map((id, position) => [id, position]));
  return [...cameras].sort((left, right) => {
    const leftIndex = index.get(left.id);
    const rightIndex = index.get(right.id);
    if (leftIndex === undefined && rightIndex === undefined) return 0;
    if (leftIndex === undefined) return 1;
    if (rightIndex === undefined) return -1;
    return leftIndex - rightIndex;
  });
}

const EMPTY_FORM = {
  id: "",
  name: "",
  location: "",
  source_type: "webcam" as "rtsp" | "file" | "webcam",
  source_uri: "0",
  enabled: true,
  inference_fps: 10,
  threshold: 0.8,
  sensitivity: "balanced" as "low" | "balanced" | "high",
};
export function CamerasPage() {
  const [cameras, setCameras] = useState<ManagedCamera[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [prefs, setPrefs] = useState<CameraPrefsMap>(loadPrefs);
  const [modalOpen, setModalOpen] = useState(false);
  const [editing, setEditing] = useState<ManagedCamera | null>(null);
  const [form, setForm] = useState({ ...EMPTY_FORM });
  const [saving, setSaving] = useState(false);
  const [toast, setToast] = useState<{ tone: "success" | "error"; title: string; description: string } | null>(null);
  const [testResult, setTestResult] = useState<{
    ok: boolean;
    error: string | null;
    preview: string | null;
  } | null>(null);
  const [testing, setTesting] = useState(false);
  const [draggedId, setDraggedId] = useState<string | null>(null);
  const [search, setSearch] = useState("");

  const loadCameras = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await getCameras();
      setCameras(sortByOrder(response.cameras, loadOrder()));
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not load cameras.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadCameras();
  }, [loadCameras]);

  const openAdd = () => {
    setEditing(null);
    setForm({ ...EMPTY_FORM });
    setTestResult(null);
    setModalOpen(true);
  };

  const openEdit = (camera: ManagedCamera) => {
    const preference = prefs[camera.id];
    setEditing(camera);
    setForm({
      id: camera.id,
      name: camera.name,
      location: camera.location,
      source_type: camera.source_type === "rtsp" ? "rtsp" : camera.source_type === "file" ? "file" : "webcam",
      source_uri: camera.source_uri,
      enabled: camera.enabled,
      inference_fps: 10,
      threshold: preference?.threshold ?? 0.8,
      sensitivity: preference?.sensitivity ?? "balanced",
    });
    setTestResult(null);
    setModalOpen(true);
  };

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    setSaving(true);
    try {
      const payload = {
        id: form.id.trim() || undefined,
        name: form.name.trim(),
        location: form.location.trim(),
        source_type: form.source_type,
        source_uri: form.source_uri.trim(),
        enabled: form.enabled,
        inference_fps: form.inference_fps,
      };
      let saved: ManagedCamera;
      if (editing) {
        saved = await updateCamera(editing.id, {
          name: payload.name,
          location: payload.location,
          source_type: payload.source_type,
          source_uri: payload.source_uri,
          enabled: payload.enabled,
          inference_fps: payload.inference_fps,
        });
      } else {
        saved = await createCamera(payload);
      }
      const nextPrefs = { ...prefs, [saved.id]: { threshold: form.threshold, sensitivity: form.sensitivity } };
      setPrefs(nextPrefs);
      savePrefs(nextPrefs);
      setModalOpen(false);
      setToast({ tone: "success", title: "Camera saved", description: `${saved.name} is ready.` });
      await loadCameras();
    } catch (requestError) {
      setToast({
        tone: "error",
        title: "Could not save camera",
        description: requestError instanceof Error ? requestError.message : "Request failed.",
      });
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async (camera: ManagedCamera) => {
    if (!window.confirm(`Delete ${camera.name}?`)) return;
    try {
      await deleteCamera(camera.id);
      setCameras((current) => current.filter((item) => item.id !== camera.id));
      setToast({ tone: "success", title: "Camera deleted", description: `${camera.name} removed.` });
    } catch (requestError) {
      setToast({
        tone: "error",
        title: "Could not delete camera",
        description: requestError instanceof Error ? requestError.message : "Request failed.",
      });
    }
  };

  const handleToggle = async (camera: ManagedCamera) => {
    try {
      const updated = await updateCamera(camera.id, { enabled: !camera.enabled });
      setCameras((current) =>
        current.map((item) => (item.id === updated.id ? { ...item, ...updated } : item)),
      );
    } catch (requestError) {
      setToast({
        tone: "error",
        title: "Could not update camera",
        description: requestError instanceof Error ? requestError.message : "Request failed.",
      });
    }
  };

  const handleStartStop = async (camera: ManagedCamera, start: boolean) => {
    try {
      const updated = start ? await startCamera(camera.id) : await stopCamera(camera.id);
      setCameras((current) =>
        current.map((item) => (item.id === updated.id ? { ...item, ...updated } : item)),
      );
    } catch (requestError) {
      setToast({
        tone: "error",
        title: start ? "Could not start camera" : "Could not stop camera",
        description: requestError instanceof Error ? requestError.message : "Request failed.",
      });
    }
  };
  const handleTest = async (cameraId: string) => {
    setTesting(true);
    setTestResult(null);
    try {
      const result = await testCamera(cameraId);
      setTestResult({
        ok: result.ok,
        error: result.error,
        preview: result.preview_frame_base64
          ? `data:${result.preview_mime_type ?? "image/jpeg"};base64,${result.preview_frame_base64}`
          : null,
      });
    } catch (requestError) {
      setTestResult({
        ok: false,
        error: requestError instanceof Error ? requestError.message : "Test failed.",
        preview: null,
      });
    } finally {
      setTesting(false);
    }
  };

  const handleDrop = (targetId: string) => {
    if (!draggedId || draggedId === targetId) return;
    setCameras((current) => {
      const next = [...current];
      const from = next.findIndex((camera) => camera.id === draggedId);
      const to = next.findIndex((camera) => camera.id === targetId);
      if (from < 0 || to < 0) return current;
      const [moved] = next.splice(from, 1);
      next.splice(to, 0, moved);
      saveOrder(next.map((camera) => camera.id));
      return next;
    });
    setDraggedId(null);
  };

  const filtered = cameras.filter((camera) => {
    const query = search.trim().toLowerCase();
    if (!query) return true;
    return (
      camera.name.toLowerCase().includes(query) ||
      camera.id.toLowerCase().includes(query) ||
      camera.location.toLowerCase().includes(query)
    );
  });

  return (
    <div className="page-container cameras-page">
      <PageIntro
        eyebrow="Administration"
        title="Manage cameras."
        description="Add, edit, test, enable, and reorder the cameras used by Mob-Alert."
        action={
          <Button onClick={openAdd} icon={<Plus size={16} aria-hidden="true" />}>
            Add camera
          </Button>
        }
      />

      {toast && (
        <Toast
          tone={toast.tone}
          title={toast.title}
          description={toast.description}
          onDismiss={() => setToast(null)}
        />
      )}

      <Card className="cameras-toolbar">
        <div className="cameras-search">
          <Search size={16} aria-hidden="true" />
          <input
            type="search"
            placeholder="Search cameras"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            aria-label="Search cameras"
          />
        </div>
        <Badge tone="neutral">{cameras.length} cameras</Badge>
      </Card>

      {loading ? (
        <div className="cameras-skeleton" aria-label="Loading cameras">
          {Array.from({ length: 4 }, (_, index) => (
            <Card className="camera-admin-card" key={index}>
              <span className="skeleton-block skeleton-camera-title" />
              <span className="skeleton-block skeleton-camera-meta" />
              <span className="skeleton-block skeleton-camera-actions" />
            </Card>
          ))}
        </div>
      ) : error ? (
        <Card className="cameras-empty">
          <p>{error}</p>
          <Button variant="secondary" onClick={() => void loadCameras()}>
            Retry
          </Button>
        </Card>
      ) : filtered.length === 0 ? (
        <Card className="cameras-empty">
          <CameraIcon size={30} aria-hidden="true" />
          <h2>No cameras yet</h2>
          <p>Add a camera to start monitoring.</p>
          <Button onClick={openAdd} icon={<Plus size={16} aria-hidden="true" />}>
            Add camera
          </Button>
        </Card>
      ) : (
        <div className="camera-admin-list">
          {filtered.map((camera) => {
            const preference = prefs[camera.id];
            return (
              <Card
                className="camera-admin-card"
                key={camera.id}
                draggable
                onDragStart={() => setDraggedId(camera.id)}
                onDragOver={(event) => event.preventDefault()}
                onDrop={() => handleDrop(camera.id)}
              >
                <div className="camera-admin-handle" aria-hidden="true">
                  <GripVertical size={18} />
                </div>
                <div className="camera-admin-main">
                  <div className="camera-admin-title-row">
                    <h3>{camera.name}</h3>
                    <Badge tone={camera.enabled ? "brand" : "neutral"}>
                      {camera.enabled ? "Enabled" : "Disabled"}
                    </Badge>
                    {camera.online ? (
                      <Badge tone="success">Online</Badge>
                    ) : (
                      <Badge tone="danger">Offline</Badge>
                    )}
                  </div>
                  <p className="camera-admin-meta">
                    {camera.id} · {camera.location || "No location"} · {camera.source_type} · {camera.source_uri}
                  </p>
                  <div className="camera-admin-prefs">
                    <span>Threshold {Math.round((preference?.threshold ?? 0.8) * 100)}%</span>
                    <span>Sensitivity {preference?.sensitivity ?? "balanced"}</span>
                  </div>
                </div>
                <div className="camera-admin-actions">
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => void handleToggle(camera)}
                    icon={<Power size={14} aria-hidden="true" />}
                  >
                    {camera.enabled ? "Disable" : "Enable"}
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => void handleStartStop(camera, !camera.online)}
                  >
                    {camera.online ? "Stop" : "Start"}
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => void handleTest(camera.id)}
                    icon={<CheckCircle2 size={14} aria-hidden="true" />}
                  >
                    Test
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => openEdit(camera)}
                    icon={<Pencil size={14} aria-hidden="true" />}
                  >
                    Edit
                  </Button>
                  <Button
                    variant="danger"
                    size="sm"
                    onClick={() => void handleDelete(camera)}
                    icon={<Trash2 size={14} aria-hidden="true" />}
                  >
                    Delete
                  </Button>
                </div>
              </Card>
            );
          })}
        </div>
      )}

      {testResult && (
        <Card className="camera-test-result">
          <div className="camera-test-header">
            <h3>Connection test</h3>
            <Button variant="ghost" size="sm" onClick={() => setTestResult(null)} icon={<X size={14} />}>
              Close
            </Button>
          </div>
          {testResult.ok ? (
            <>
              <p className="camera-test-ok">Connection succeeded.</p>
              {testResult.preview && (
                <img src={testResult.preview} alt="Camera preview" className="camera-test-preview" />
              )}
            </>
          ) : (
            <p className="camera-test-error">{testResult.error ?? "Connection failed."}</p>
          )}
        </Card>
      )}

      <Modal
        open={modalOpen}
        onClose={() => setModalOpen(false)}
        title={editing ? "Edit camera" : "Add camera"}
      >
        <form className="camera-form" onSubmit={handleSubmit}>
          {!editing && (
            <label>
              Camera ID
              <input
                value={form.id}
                onChange={(event) => setForm((current) => ({ ...current, id: event.target.value }))}
                placeholder="lobby"
              />
            </label>
          )}
          <label>
            Name
            <input
              required
              value={form.name}
              onChange={(event) => setForm((current) => ({ ...current, name: event.target.value }))}
            />
          </label>
          <label>
            Location
            <input
              value={form.location}
              onChange={(event) => setForm((current) => ({ ...current, location: event.target.value }))}
            />
          </label>
          <label>
            Source type
            <select
              value={form.source_type}
              onChange={(event) =>
                setForm((current) => ({
                  ...current,
                  source_type: event.target.value as "rtsp" | "file" | "webcam",
                }))
              }
            >
              <option value="webcam">Webcam</option>
              <option value="rtsp">RTSP / URL</option>
              <option value="file">Video file</option>
            </select>
          </label>
          <label>
            Source URI
            <input
              required
              value={form.source_uri}
              onChange={(event) => setForm((current) => ({ ...current, source_uri: event.target.value }))}
              placeholder="0 or rtsp://... or C:\\video.mp4"
            />
          </label>
          <label>
            Inference FPS
            <input
              type="number"
              min="0.1"
              step="0.1"
              value={form.inference_fps}
              onChange={(event) =>
                setForm((current) => ({ ...current, inference_fps: Number(event.target.value) }))
              }
            />
          </label>
          <label>
            Alert threshold
            <input
              type="range"
              min="0.3"
              max="0.95"
              step="0.05"
              value={form.threshold}
              onChange={(event) =>
                setForm((current) => ({ ...current, threshold: Number(event.target.value) }))
              }
            />
            <span>{Math.round(form.threshold * 100)}%</span>
          </label>
          <label>
            Sensitivity
            <select
              value={form.sensitivity}
              onChange={(event) =>
                setForm((current) => ({
                  ...current,
                  sensitivity: event.target.value as "low" | "balanced" | "high",
                }))
              }
            >
              <option value="low">Low</option>
              <option value="balanced">Balanced</option>
              <option value="high">High</option>
            </select>
          </label>
          <label className="camera-enabled-toggle">
            <input
              type="checkbox"
              checked={form.enabled}
              onChange={(event) => setForm((current) => ({ ...current, enabled: event.target.checked }))}
            />
            Enabled
          </label>
          <div className="camera-form-actions">
            {editing && (
              <Button
                type="button"
                variant="secondary"
                onClick={() => void handleTest(editing.id)}
                disabled={testing}
              >
                {testing ? <Spinner size="sm" /> : "Test connection"}
              </Button>
            )}
            <Button type="submit" disabled={saving}>
              {saving ? <Spinner size="sm" /> : editing ? "Save changes" : "Create camera"}
            </Button>
          </div>
          {editing && testResult && (
            <div className="camera-test-inline">
              {testResult.ok ? (
                <>
                  <p className="camera-test-ok">Connection succeeded.</p>
                  {testResult.preview && (
                    <img src={testResult.preview} alt="Camera preview" className="camera-test-preview" />
                  )}
                </>
              ) : (
                <p className="camera-test-error">{testResult.error ?? "Connection failed."}</p>
              )}
            </div>
          )}
        </form>
      </Modal>
    </div>
  );
}

