import {
  AlertTriangle,
  CalendarDays,
  Camera,
  ChevronLeft,
  ChevronRight,
  Download,
  Eye,
  Filter,
  ShieldCheck,
  Video,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  acknowledgeIncident,
  getCameras,
  getIncidents,
  getIncidentsCsvUrl,
  markIncidentFalsePositive,
  type Incident,
  type ManagedCamera,
} from "../api/client";
import { API_URL } from "../api/client";
import { Badge } from "../components/Badge";
import { Button } from "../components/Button";
import { Card } from "../components/Card";
import { Modal } from "../components/Modal";
import { PageIntro } from "../components/PageIntro";
import { Toast } from "../components/Toast";

const PAGE_SIZE = 12;

function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return timestamp;
  return date.toLocaleString([], {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function durationLabel(start: string, end: string | null): string {
  if (!end) return "Open";
  const startDate = new Date(start);
  const endDate = new Date(end);
  if (Number.isNaN(startDate.getTime()) || Number.isNaN(endDate.getTime())) return "—";
  const seconds = Math.max(0, (endDate.getTime() - startDate.getTime()) / 1000);
  if (seconds < 60) return `${Math.round(seconds)}s`;
  return `${Math.round(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

function statusTone(status: Incident["status"]): "danger" | "warning" | "success" | "neutral" {
  if (status === "new") return "danger";
  if (status === "acknowledged") return "success";
  return "neutral";
}
export function IncidentsPage() {
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [cameras, setCameras] = useState<ManagedCamera[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(0);
  const [page, setPage] = useState(1);
  const [cameraId, setCameraId] = useState("");
  const [status, setStatus] = useState<"" | "new" | "acknowledged" | "false-positive">("");
  const [search, setSearch] = useState("");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [selected, setSelected] = useState<Incident | null>(null);
  const [toast, setToast] = useState<{ tone: "success" | "error"; title: string; description: string } | null>(null);
  const [busyId, setBusyId] = useState<number | null>(null);

  const loadCameras = useCallback(async () => {
    try {
      const response = await getCameras();
      setCameras(response.cameras);
    } catch {
      // Keep incident workflow usable even if camera list fails.
    }
  }, []);

  const loadIncidents = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await getIncidents({
        page,
        pageSize: PAGE_SIZE,
        cameraId: cameraId || undefined,
        status: status || undefined,
        startTime: startDate ? new Date(startDate).toISOString() : undefined,
        endTime: endDate ? new Date(`${endDate}T23:59:59`).toISOString() : undefined,
        search: search.trim() || undefined,
      });
      setIncidents(response.incidents);
      setTotal(response.total);
      setTotalPages(response.total_pages);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not load incidents.");
    } finally {
      setLoading(false);
    }
  }, [cameraId, endDate, page, search, startDate, status]);

  useEffect(() => {
    void loadCameras();
  }, [loadCameras]);

  useEffect(() => {
    void loadIncidents();
  }, [loadIncidents]);

  const todayCount = useMemo(() => {
    const today = new Date().toDateString();
    return incidents.filter((incident) => new Date(incident.start_time).toDateString() === today).length;
  }, [incidents]);

  const perCamera = useMemo(() => {
    const counts = new Map<string, number>();
    incidents.forEach((incident) => {
      counts.set(incident.camera_id, (counts.get(incident.camera_id) ?? 0) + 1);
    });
    if (counts.size === 0) return 0;
    return Math.round((incidents.length / counts.size) * 10) / 10;
  }, [incidents]);

  const averageResponseTime = useMemo(() => {
    const durations = incidents
      .filter((incident) => incident.end_time)
      .map((incident) => {
        const start = new Date(incident.start_time).getTime();
        const end = new Date(incident.end_time as string).getTime();
        return Math.max(0, end - start) / 1000;
      });
    if (durations.length === 0) return "—";
    const average = durations.reduce((sum, value) => sum + value, 0) / durations.length;
    if (average < 60) return `${Math.round(average)}s`;
    return `${Math.round(average / 60)}m`;
  }, [incidents]);

  const cameraName = (cameraIdValue: string) =>
    cameras.find((camera) => camera.id === cameraIdValue)?.name ?? cameraIdValue;

  const handleAcknowledge = async (incident: Incident) => {
    setBusyId(incident.id);
    try {
      const updated = await acknowledgeIncident(incident.id);
      setIncidents((current) => current.map((item) => (item.id === updated.id ? updated : item)));
      if (selected?.id === updated.id) setSelected(updated);
      setToast({ tone: "success", title: "Incident acknowledged", description: `Incident ${incident.id} updated.` });
    } catch (requestError) {
      setToast({
        tone: "error",
        title: "Could not acknowledge incident",
        description: requestError instanceof Error ? requestError.message : "Request failed.",
      });
    } finally {
      setBusyId(null);
    }
  };

  const handleFalsePositive = async (incident: Incident) => {
    setBusyId(incident.id);
    try {
      const updated = await markIncidentFalsePositive(incident.id);
      setIncidents((current) => current.map((item) => (item.id === updated.id ? updated : item)));
      if (selected?.id === updated.id) setSelected(updated);
      setToast({ tone: "success", title: "Marked false positive", description: `Incident ${incident.id} updated.` });
    } catch (requestError) {
      setToast({
        tone: "error",
        title: "Could not update incident",
        description: requestError instanceof Error ? requestError.message : "Request failed.",
      });
    } finally {
      setBusyId(null);
    }
  };

  const exportUrl = getIncidentsCsvUrl({
    cameraId: cameraId || undefined,
    status: status || undefined,
    startTime: startDate ? new Date(startDate).toISOString() : undefined,
    endTime: endDate ? new Date(`${endDate}T23:59:59`).toISOString() : undefined,
    search: search.trim() || undefined,
  });

  return (
    <div className="page-container incidents-page">
      <PageIntro
        eyebrow="Review"
        title="Incidents."
        description="Review persisted incidents, filter evidence, and export reports."
        action={
          <a className="button-link" href={exportUrl} target="_blank" rel="noreferrer">
            <Button icon={<Download size={16} aria-hidden="true" />}>Export CSV</Button>
          </a>
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

      <div className="incidents-stats-grid">
        <Card className="incidents-stat-card">
          <span className="incidents-stat-icon"><AlertTriangle size={18} aria-hidden="true" /></span>
          <div>
            <p className="status-label">Incidents today</p>
            <p className="status-value">{todayCount}</p>
          </div>
        </Card>
        <Card className="incidents-stat-card">
          <span className="incidents-stat-icon"><Camera size={18} aria-hidden="true" /></span>
          <div>
            <p className="status-label">Per camera (avg)</p>
            <p className="status-value">{perCamera}</p>
          </div>
        </Card>
        <Card className="incidents-stat-card">
          <span className="incidents-stat-icon"><CalendarDays size={18} aria-hidden="true" /></span>
          <div>
            <p className="status-label">Avg. response time</p>
            <p className="status-value">{averageResponseTime}</p>
          </div>
        </Card>
      </div>

      <Card className="incidents-filters">
        <div className="incidents-filter-row">
          <label>
            Camera
            <select value={cameraId} onChange={(event) => { setCameraId(event.target.value); setPage(1); }}>
              <option value="">All cameras</option>
              {cameras.map((camera) => (
                <option key={camera.id} value={camera.id}>
                  {camera.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Status
            <select
              value={status}
              onChange={(event) => {
                setStatus(event.target.value as "" | "new" | "acknowledged" | "false-positive");
                setPage(1);
              }}
            >
              <option value="">All statuses</option>
              <option value="new">New</option>
              <option value="acknowledged">Acknowledged</option>
              <option value="false-positive">False positive</option>
            </select>
          </label>
          <label>
            Start date
            <input type="date" value={startDate} onChange={(event) => { setStartDate(event.target.value); setPage(1); }} />
          </label>
          <label>
            End date
            <input type="date" value={endDate} onChange={(event) => { setEndDate(event.target.value); setPage(1); }} />
          </label>
          <label className="incidents-search">
            Search
            <div className="cameras-search">
              <Filter size={16} aria-hidden="true" />
              <input
                type="search"
                placeholder="Camera ID"
                value={search}
                onChange={(event) => { setSearch(event.target.value); setPage(1); }}
              />
            </div>
          </label>
        </div>
      </Card>

      {loading ? (
        <div className="incidents-skeleton" aria-label="Loading incidents">
          {Array.from({ length: 6 }, (_, index) => (
            <Card className="incidents-skeleton-row" key={index}>
              <span className="skeleton-block skeleton-incident" />
            </Card>
          ))}
        </div>
      ) : error ? (
        <Card className="incidents-empty">
          <p>{error}</p>
          <Button variant="secondary" onClick={() => void loadIncidents()}>
            Retry
          </Button>
        </Card>
      ) : incidents.length === 0 ? (
        <Card className="incidents-empty">
          <ShieldCheck size={30} aria-hidden="true" />
          <h2>No incidents found</h2>
          <p>Adjust your filters or wait for new alerts.</p>
        </Card>
      ) : (
        <>
          <div className="incidents-table-wrap">
            <table className="incidents-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Camera</th>
                  <th>Started</th>
                  <th>Ended</th>
                  <th>Peak</th>
                  <th>Status</th>
                  <th>Duration</th>
                  <th>Evidence</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {incidents.map((incident) => (
                  <tr key={incident.id}>
                    <td>{incident.id}</td>
                    <td>{cameraName(incident.camera_id)}</td>
                    <td>{formatTimestamp(incident.start_time)}</td>
                    <td>{incident.end_time ? formatTimestamp(incident.end_time) : "Open"}</td>
                    <td>{Math.round(incident.peak_confidence * 100)}%</td>
                    <td>
                      <Badge tone={statusTone(incident.status)}>{incident.status}</Badge>
                    </td>
                    <td>{durationLabel(incident.start_time, incident.end_time)}</td>
                    <td>
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={() => setSelected(incident)}
                        icon={<Eye size={14} aria-hidden="true" />}
                      >
                        View
                      </Button>
                    </td>
                    <td className="incidents-actions">
                      {incident.status === "new" && (
                        <>
                          <Button
                            variant="secondary"
                            size="sm"
                            disabled={busyId === incident.id}
                            onClick={() => void handleAcknowledge(incident)}
                          >
                            Acknowledge
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            disabled={busyId === incident.id}
                            onClick={() => void handleFalsePositive(incident)}
                          >
                            False positive
                          </Button>
                        </>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="incidents-table-footer">
            <p>
              Showing <strong>{incidents.length}</strong> of <strong>{total}</strong>
            </p>
            <div className="alerts-pagination">
              <Button
                variant="secondary"
                size="sm"
                onClick={() => setPage((current) => Math.max(1, current - 1))}
                disabled={page <= 1}
                icon={<ChevronLeft size={15} aria-hidden="true" />}
              >
                Previous
              </Button>
              <span>
                Page <strong>{page}</strong> of <strong>{Math.max(1, totalPages)}</strong>
              </span>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => setPage((current) => Math.min(totalPages, current + 1))}
                disabled={page >= totalPages}
                icon={<ChevronRight size={15} aria-hidden="true" />}
              >
                Next
              </Button>
            </div>
          </div>
        </>
      )}

      <Modal open={selected !== null} onClose={() => setSelected(null)} title={selected ? `Incident ${selected.id}` : "Incident"}>
        {selected && (
          <div className="incident-evidence">
            <p>
              <strong>Camera:</strong> {cameraName(selected.camera_id)} · <strong>Status:</strong> {selected.status}
            </p>
            <p>
              <strong>Started:</strong> {formatTimestamp(selected.start_time)} · <strong>Peak:</strong>{" "}
              {Math.round(selected.peak_confidence * 100)}%
            </p>
            <div className="incident-evidence-media">
              {selected.snapshot_url ? (
                <img
                  src={`${API_URL}${selected.snapshot_url}`}
                  alt={`Incident ${selected.id} snapshot`}
                  className="incident-snapshot"
                />
              ) : (
                <div className="incident-media-empty">
                  <Camera size={24} aria-hidden="true" />
                  <span>No snapshot</span>
                </div>
              )}
              {selected.clip_url ? (
                <video controls className="incident-video" src={`${API_URL}${selected.clip_url}`} />
              ) : (
                <div className="incident-media-empty">
                  <Video size={24} aria-hidden="true" />
                  <span>No clip</span>
                </div>
              )}
            </div>
            {selected.status === "new" && (
              <div className="incident-evidence-actions">
                <Button
                  onClick={() => void handleAcknowledge(selected)}
                  disabled={busyId === selected.id}
                >
                  Acknowledge
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => void handleFalsePositive(selected)}
                  disabled={busyId === selected.id}
                >
                  False positive
                </Button>
              </div>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
}

