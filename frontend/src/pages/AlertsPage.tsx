import {
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  Bell,
  Camera,
  ChevronLeft,
  ChevronRight,
  Clock3,
  Download,
  FileVideo2,
  Search,
  ShieldCheck,
  SlidersHorizontal,
} from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  getAlerts,
  getAlertsCsvUrl,
  getAlertSnapshotUrl,
  type AlertStats,
  type LiveAlert,
} from "../api/client";
import { Badge } from "../components/Badge";
import { Button } from "../components/Button";
import { Card } from "../components/Card";
import { PageIntro } from "../components/PageIntro";
import { Toast } from "../components/Toast";

type SourceFilter = "all" | "live" | "upload";
const PAGE_SIZE = 10;

const EMPTY_STATS: AlertStats = {
  total_alerts: 0,
  last_24h: 0,
  average_confidence: 0,
};

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

function formatConfidence(value: number): string {
  return `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
}

function SnapshotThumbnail({ alert }: { alert: LiveAlert }) {
  const [failed, setFailed] = useState(false);
  if (!alert.snapshot_url || failed) {
    return (
      <span className="alert-snapshot-empty" aria-label="Snapshot not available">
        <Camera size={16} aria-hidden="true" />
      </span>
    );
  }
  return (
    <a
      className="alert-snapshot-link"
      href={getAlertSnapshotUrl(alert.snapshot_url)}
      target="_blank"
      rel="noreferrer"
      aria-label={`Open snapshot for ${alert.source_name}`}
    >
      <img
        src={getAlertSnapshotUrl(alert.snapshot_url)}
        alt={`Alert snapshot for ${alert.source_name}`}
        loading="lazy"
        onError={() => setFailed(true)}
      />
    </a>
  );
}

function AlertTableSkeleton() {
  return (
    <div className="alerts-skeleton-table" aria-label="Loading alerts">
      {Array.from({ length: 6 }, (_, index) => (
        <div className="alerts-skeleton-row" key={index}>
          <span className="skeleton-block skeleton-time" />
          <span className="skeleton-block skeleton-source" />
          <span className="skeleton-block skeleton-confidence" />
          <span className="skeleton-block skeleton-image" />
        </div>
      ))}
    </div>
  );
}

function StatsSkeleton() {
  return (
    <div className="alerts-stats-grid" aria-label="Loading alert statistics">
      {Array.from({ length: 3 }, (_, index) => (
        <Card className="alerts-stat-card" key={index}>
          <span className="skeleton-block skeleton-stat-icon" />
          <span className="skeleton-block skeleton-stat-text" />
          <span className="skeleton-block skeleton-stat-number" />
        </Card>
      ))}
    </div>
  );
}

export function AlertsPage() {
  const [alerts, setAlerts] = useState<LiveAlert[]>([]);
  const [stats, setStats] = useState(EMPTY_STATS);
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(0);
  const [page, setPage] = useState(1);
  const [source, setSource] = useState<SourceFilter>("all");
  const [searchInput, setSearchInput] = useState("");
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setPage(1);
      setSearch(searchInput.trim());
    }, 250);
    return () => window.clearTimeout(timer);
  }, [searchInput]);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    void getAlerts({
      page,
      pageSize: PAGE_SIZE,
      source: source === "all" ? undefined : source,
      search: search || undefined,
      signal: controller.signal,
    })
      .then((response) => {
        setAlerts(response.alerts);
        setStats(response.stats);
        setTotal(response.total);
        setTotalPages(response.total_pages);
      })
      .catch((requestError: unknown) => {
        if (controller.signal.aborted) return;
        setError(
          requestError instanceof Error
            ? requestError.message
            : "Could not load alert history.",
        );
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [page, search, source, reloadKey]);

  function resetFilters() {
    setSource("all");
    setSearchInput("");
    setSearch("");
    setPage(1);
  }

  function handleSourceChange(nextSource: SourceFilter) {
    setSource(nextSource);
    setPage(1);
  }

  function handleExport() {
    const link = document.createElement("a");
    link.href = getAlertsCsvUrl({
      source: source === "all" ? undefined : source,
      search: search || undefined,
    });
    link.download = "alerts.csv";
    document.body.append(link);
    link.click();
    link.remove();
  }

  const rangeStart = total === 0 ? 0 : (page - 1) * PAGE_SIZE + 1;
  const rangeEnd = Math.min(page * PAGE_SIZE, total);
  const hasFilters = source !== "all" || search.length > 0;

  return (
    <div className="page-container alerts-page">
      <PageIntro
        eyebrow="Event history"
        title="Alerts, in one place."
        description="Review persisted screening events from live camera sessions and uploaded videos."
        action={
          <Button
            size="sm"
            variant="secondary"
            icon={<Download size={16} aria-hidden="true" />}
            onClick={handleExport}
            disabled={loading || total === 0}
          >
            Export CSV
          </Button>
        }
      />

      {error && (
        <Toast
          tone="error"
          title="Could not load alert history"
          description={error}
          onDismiss={() => setError(null)}
        />
      )}

      {loading ? (
        <StatsSkeleton />
      ) : (
        <div className="alerts-stats-grid">
          <Card className="alerts-stat-card">
            <span className="alerts-stat-icon alerts-stat-icon-brand">
              <Bell size={18} aria-hidden="true" />
            </span>
            <div className="alerts-stat-copy">
              <p>Total alerts</p>
              <strong>{stats.total_alerts.toLocaleString()}</strong>
            </div>
            <span className="alerts-stat-decoration" aria-hidden="true">
              <ArrowUp size={15} />
            </span>
          </Card>
          <Card className="alerts-stat-card">
            <span className="alerts-stat-icon alerts-stat-icon-warning">
              <Clock3 size={18} aria-hidden="true" />
            </span>
            <div className="alerts-stat-copy">
              <p>Last 24 hours</p>
              <strong>{stats.last_24h.toLocaleString()}</strong>
            </div>
            <span className="alerts-stat-note">rolling</span>
          </Card>
          <Card className="alerts-stat-card">
            <span className="alerts-stat-icon alerts-stat-icon-danger">
              <AlertTriangle size={18} aria-hidden="true" />
            </span>
            <div className="alerts-stat-copy">
              <p>Average confidence</p>
              <strong>{formatConfidence(stats.average_confidence)}</strong>
            </div>
            <span className="alerts-stat-note">all time</span>
          </Card>
        </div>
      )}

      <Card className="alerts-history-card">
        <div className="alerts-toolbar">
          <div className="alerts-toolbar-title">
            <span className="alerts-toolbar-icon">
              <SlidersHorizontal size={16} aria-hidden="true" />
            </span>
            <div>
              <h2>Alert history</h2>
              <p>{loading ? "Loading recent events…" : `${total.toLocaleString()} matching events`}</p>
            </div>
          </div>
          <div className="alerts-filters">
            <label className="alerts-search">
              <Search size={15} aria-hidden="true" />
              <span className="sr-only">Search alert source or label</span>
              <input
                type="search"
                placeholder="Search alerts…"
                value={searchInput}
                onChange={(event) => setSearchInput(event.target.value)}
              />
            </label>
            <label className="alerts-source-select">
              <span className="sr-only">Filter by source</span>
              <select
                value={source}
                onChange={(event) =>
                  handleSourceChange(event.target.value as SourceFilter)
                }
              >
                <option value="all">All sources</option>
                <option value="live">Live</option>
                <option value="upload">Upload</option>
              </select>
            </label>
          </div>
        </div>

        {loading ? (
          <AlertTableSkeleton />
        ) : error ? (
          <div className="alerts-load-error">
            <ShieldCheck size={23} aria-hidden="true" />
            <p>Alert history could not be loaded.</p>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setReloadKey((key) => key + 1)}
            >
              Try again
            </Button>
          </div>
        ) : alerts.length === 0 ? (
          hasFilters ? (
            <div className="alerts-filter-empty">
              <span className="alerts-empty-icon">
                <Search size={20} aria-hidden="true" />
              </span>
              <h3>No matching alerts</h3>
              <p>Try a different search term or clear your filters.</p>
              <Button variant="secondary" size="sm" onClick={resetFilters}>
                Clear filters
              </Button>
            </div>
          ) : (
            <div className="alerts-empty-state">
              <div className="alerts-empty-art" aria-hidden="true">
                <span className="alerts-empty-orbit alerts-empty-orbit-one" />
                <span className="alerts-empty-orbit alerts-empty-orbit-two" />
                <span className="alerts-empty-icon alerts-empty-icon-large">
                  <ShieldCheck size={25} />
                </span>
                <span className="alerts-empty-spark alerts-empty-spark-one" />
                <span className="alerts-empty-spark alerts-empty-spark-two" />
              </div>
              <Badge tone="success">All clear for now</Badge>
              <h3>Your alert history is empty</h3>
              <p>
                When the screening model raises an alert, it will be saved here
                with its confidence and snapshot for review.
              </p>
              <div className="alerts-empty-actions">
                <Link to="/live" className="button button-primary button-md">
                  <Camera size={15} aria-hidden="true" />
                  Live screening
                </Link>
                <Link to="/upload" className="button button-secondary button-md">
                  <FileVideo2 size={15} aria-hidden="true" />
                  Analyze a video
                </Link>
              </div>
            </div>
          )
        ) : (
          <>
            <div className="alerts-table-scroll">
              <table className="alerts-table">
                <thead>
                  <tr>
                    <th scope="col">Time</th>
                    <th scope="col">Source</th>
                    <th scope="col">Confidence</th>
                    <th scope="col">Snapshot</th>
                  </tr>
                </thead>
                <tbody>
                  {alerts.map((alert) => (
                    <tr key={alert.id}>
                      <td>
                        <time dateTime={alert.timestamp}>
                          {formatTimestamp(alert.timestamp)}
                        </time>
                        {alert.video_time_seconds !== null && (
                          <span className="alerts-video-time">
                            Video at {Math.floor(alert.video_time_seconds / 60)}:
                            {String(Math.floor(alert.video_time_seconds % 60)).padStart(2, "0")}
                          </span>
                        )}
                      </td>
                      <td>
                        <div className="alerts-source-cell">
                          <Badge tone={alert.source === "live" ? "brand" : "neutral"}>
                            {alert.source === "live" ? (
                              <Camera size={12} aria-hidden="true" />
                            ) : (
                              <FileVideo2 size={12} aria-hidden="true" />
                            )}
                            {alert.source === "live" ? "Live" : "Upload"}
                          </Badge>
                          <span className="alerts-source-name" title={alert.source_name}>
                            {alert.source_name}
                          </span>
                        </div>
                      </td>
                      <td>
                        <div className="alerts-confidence-cell">
                          <span className="alerts-confidence-value">
                            {formatConfidence(alert.smoothed_confidence)}
                          </span>
                          <span
                            className="alerts-confidence-track"
                            aria-hidden="true"
                          >
                            <span
                              style={{
                                width: `${Math.min(100, Math.max(0, alert.smoothed_confidence * 100))}%`,
                              }}
                            />
                          </span>
                        </div>
                      </td>
                      <td>
                        <SnapshotThumbnail alert={alert} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="alerts-table-footer">
              <p>
                Showing <strong>{rangeStart}–{rangeEnd}</strong> of{" "}
                <strong>{total}</strong>
              </p>
              <div className="alerts-pagination">
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => setPage((current) => Math.max(1, current - 1))}
                  disabled={page <= 1}
                  aria-label="Previous page"
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
                  aria-label="Next page"
                  icon={<ChevronRight size={15} aria-hidden="true" />}
                >
                  Next
                </Button>
              </div>
            </div>
          </>
        )}
      </Card>

      <div className="alerts-disclaimer">
        <ShieldCheck size={17} aria-hidden="true" />
        <p>
          A screening alert is only a signal for review. It does not confirm
          that violence occurred.
        </p>
        <ArrowDown size={14} className="alerts-disclaimer-mark" aria-hidden="true" />
      </div>
    </div>
  );
}
