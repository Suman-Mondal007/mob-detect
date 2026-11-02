from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_path: Path = Path("models/best.pt")
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    port: int = Field(default=8000, ge=1, le=65535)
    image_size: int = Field(default=224, gt=0)
    violence_threshold: float = Field(default=0.8, ge=0, le=1)
    max_upload_size_mb: int = Field(default=10, gt=0)
    max_image_pixels: int = Field(default=20_000_000, gt=0)
    predict_timeout_seconds: float = Field(default=60, gt=0)
    max_video_upload_size_mb: int = Field(default=500, gt=0)
    video_sampling_fps: float = Field(default=1, gt=0, le=30)
    video_window_size: int = Field(default=16, gt=0)
    video_alert_duration_seconds: float = Field(default=2, ge=0)
    max_video_duration_seconds: int = Field(default=600, gt=0)
    max_video_frame_pixels: int = Field(default=20_000_000, gt=0)
    max_active_video_jobs: int = Field(default=2, gt=0)
    max_stored_video_jobs: int = Field(default=100, gt=0)
    max_live_frame_size_bytes: int = Field(default=1_048_576, gt=0)
    max_live_frame_pixels: int = Field(default=2_000_000, gt=0)
    max_live_clients: int = Field(default=4, gt=0)
    camera_inference_fps: float = Field(default=3, gt=0, le=30)
    camera_total_inference_fps: float = Field(default=12, gt=0, le=120)
    max_cameras: int = Field(default=32, gt=0)
    camera_frame_width: int = Field(default=640, gt=0)
    camera_frame_height: int = Field(default=360, gt=0)
    camera_display_frame_width: int = Field(default=960, gt=0)
    camera_display_frame_height: int = Field(default=960, gt=0)
    incident_evidence_dir: Path = Path("outputs/incidents")
    incident_retention_days: int = Field(default=30, gt=0)
    incident_ring_buffer_seconds: float = Field(default=10, gt=0, le=60)
    incident_evidence_fps: float = Field(default=5, gt=0, le=15)
    incident_frame_width: int = Field(default=320, gt=0)
    incident_frame_height: int = Field(default=180, gt=0)
    incident_cleanup_interval_seconds: float = Field(default=3600, gt=0)
    camera_file_fps: float = Field(default=25, gt=0, le=60)
    camera_read_timeout_seconds: float = Field(default=5, gt=0, le=120)
    camera_reconnect_initial_seconds: float = Field(default=1, gt=0)
    camera_reconnect_max_seconds: float = Field(default=30, gt=0)
    camera_alert_cooldown_seconds: float = Field(default=15, ge=0, le=3600)
    alerts_database_path: Path = Path("outputs/alerts.db")
    alert_snapshots_dir: Path = Path("outputs/alert_snapshots")
    jwt_secret: str = "mob-alert-dev-secret-change-me"
    jwt_expire_minutes: int = 60
    rate_limit_per_minute: int = 120

    @property
    def resolved_model_path(self) -> Path:
        path = self.model_path.expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def max_video_upload_size_bytes(self) -> int:
        return self.max_video_upload_size_mb * 1024 * 1024

    @property
    def resolved_alerts_database_path(self) -> Path:
        path = self.alerts_database_path.expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    @property
    def resolved_alert_snapshots_dir(self) -> Path:
        path = self.alert_snapshots_dir.expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    @property
    def resolved_incident_evidence_dir(self) -> Path:
        path = self.incident_evidence_dir.expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

@lru_cache
def get_settings() -> Settings:
    return Settings()
