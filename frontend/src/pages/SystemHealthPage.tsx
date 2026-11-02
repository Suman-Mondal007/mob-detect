import { Cpu, HardDrive, HeartPulse } from "lucide-react";
import { useEffect, useState } from "react";
import { getSystemHealth } from "../api/client";
import { Badge } from "../components/Badge";
import { Card } from "../components/Card";
import { PageIntro } from "../components/PageIntro";
import { Spinner } from "../components/Spinner";

interface SystemHealth {
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
}

export function SystemHealthPage() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    getSystemHealth()
      .then((data) => {
        if (active) setHealth(data);
      })
      .catch((requestError) => {
        if (active) setError(requestError instanceof Error ? requestError.message : "Could not load system health.");
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  if (loading) {
    return (
      <div className="page-container system-health-page">
        <PageIntro eyebrow="Operations" title="System health." description="Loading system health..." />
        <Spinner />
      </div>
    );
  }

  if (error || !health) {
    return (
      <div className="page-container system-health-page">
        <PageIntro eyebrow="Operations" title="System health." description="System health is unavailable." />
        <Card className="cameras-empty">
          <p>{error ?? "No data."}</p>
        </Card>
      </div>
    );
  }

  return (
    <div className="page-container system-health-page">
      <PageIntro
        eyebrow="Operations"
        title="System health."
        description="Monitor uptime, GPU memory, and per-camera performance."
      />
      <div className="incidents-stats-grid">
        <Card className="incidents-stat-card">
          <span className="incidents-stat-icon"><HeartPulse size={18} aria-hidden="true" /></span>
          <div>
            <p className="status-label">Uptime</p>
            <p className="status-value">{Math.round(health.uptime_seconds)}s</p>
          </div>
        </Card>
        <Card className="incidents-stat-card">
          <span className="incidents-stat-icon"><Cpu size={18} aria-hidden="true" /></span>
          <div>
            <p className="status-label">GPU allocated</p>
            <p className="status-value">
              {health.gpu_memory ? Math.round(health.gpu_memory.allocated_bytes / 1024 / 1024) : 0} MB
            </p>
          </div>
        </Card>
        <Card className="incidents-stat-card">
          <span className="incidents-stat-icon"><HardDrive size={18} aria-hidden="true" /></span>
          <div>
            <p className="status-label">Cameras</p>
            <p className="status-value">{health.cameras.length}</p>
          </div>
        </Card>
      </div>
      <Card className="incidents-table-wrap">
        <table className="incidents-table">
          <thead>
            <tr>
              <th>Camera</th>
              <th>Online</th>
              <th>Input FPS</th>
              <th>Inference FPS</th>
              <th>Latency</th>
              <th>State</th>
            </tr>
          </thead>
          <tbody>
            {health.cameras.map((camera) => (
              <tr key={camera.camera_id}>
                <td>{camera.camera_id}</td>
                <td>
                  <Badge tone={camera.online ? "success" : "danger"}>
                    {camera.online ? "Online" : "Offline"}
                  </Badge>
                </td>
                <td>{camera.input_fps.toFixed(1)}</td>
                <td>{camera.inference_fps.toFixed(1)}</td>
                <td>{camera.inference_ms ? `${Math.round(camera.inference_ms)} ms` : "—"}</td>
                <td>{camera.state}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
    </div>
  );
}
