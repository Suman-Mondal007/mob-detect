import re
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    model_validator,
)


class CameraCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        default_factory=lambda: uuid4().hex,
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$",
    )
    name: str | None = Field(default=None, min_length=1, max_length=100)
    location: str = Field(default="", max_length=200)
    source_type: Literal["rtsp", "file", "webcam"] | None = None
    source_uri: Annotated[StrictStr, Field(max_length=2048)] | None = None
    enabled: StrictBool = True
    auto_start: StrictBool | None = None
    camera_id: str | None = Field(default=None, min_length=1, max_length=64)
    source: Annotated[StrictStr, Field(max_length=2048)] | StrictInt | None = None
    inference_fps: float | None = Field(
        default=None,
        gt=0,
        le=30,
        allow_inf_nan=False,
    )

    @model_validator(mode="before")
    @classmethod
    def support_legacy_camera_payload(cls, value: object) -> object:
        if not isinstance(value, dict) or "source" not in value:
            return value

        payload = dict(value)
        legacy_source = payload["source"]
        if "id" not in payload and "camera_id" in payload:
            payload["id"] = payload["camera_id"]
        if "name" not in payload and "camera_id" in payload:
            payload["name"] = payload["camera_id"]
        if "source_uri" not in payload:
            payload["source_uri"] = str(legacy_source)
        if "source_type" not in payload:
            if isinstance(legacy_source, int):
                payload["source_type"] = "webcam"
            elif isinstance(legacy_source, str) and re.match(
                r"(?i)^(rtsp|rtsps|http|https)://",
                legacy_source.strip(),
            ):
                payload["source_type"] = "rtsp"
            else:
                payload["source_type"] = "file"
        return payload

    @model_validator(mode="after")
    def require_camera_name_and_source(self) -> "CameraCreateRequest":
        if self.name is None:
            raise ValueError("Camera name is required.")
        if self.source_type is None or self.source_uri is None:
            raise ValueError("Camera source_type and source_uri are required.")
        return self


class CameraUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=100)
    location: str | None = Field(default=None, max_length=200)
    source_type: Literal["rtsp", "file", "webcam"] | None = None
    source_uri: Annotated[StrictStr, Field(max_length=2048)] | None = None
    enabled: StrictBool | None = None
    auto_start: StrictBool | None = None


class CameraPreviewResponse(BaseModel):
    ok: bool
    error: str | None = None
    preview_frame_base64: str | None = None
    preview_mime_type: Literal["image/jpeg"] | None = None


class CameraStatusResponse(BaseModel):
    camera_id: str
    source: str
    source_type: Literal["file", "url", "webcam", "rtsp"]
    online: bool
    last_frame_at: str | None = None
    last_inference_at: str | None = None
    inference_ms: float | None = Field(default=None, ge=0)
    predicted_class: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    smoothed_confidence: float | None = Field(default=None, ge=0, le=1)
    alert: bool = False
    state: Literal["NORMAL", "ALERT", "ACKNOWLEDGED", "COOLDOWN"] = "NORMAL"
    input_fps: float = Field(default=0, ge=0)
    inference_fps: float = Field(default=0, ge=0)
    error: str | None = None


class CameraListResponse(BaseModel):
    cameras: list[CameraStatusResponse]


class CameraResponse(CameraStatusResponse):
    id: str
    name: str
    location: str
    source_uri: str
    enabled: bool
    auto_start: bool = True
    created_at: str


class CameraManagementListResponse(BaseModel):
    cameras: list[CameraResponse]


class TimelinePoint(BaseModel):
    time_seconds: float = Field(ge=0)
    violence_probability: float = Field(ge=0, le=1)


class AlertRange(BaseModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)


class KeyFrame(BaseModel):
    time_seconds: float = Field(ge=0)
    violence_probability: float = Field(ge=0, le=1)
    image_base64: str


class VideoResult(BaseModel):
    verdict: str
    max_violence_probability: float = Field(ge=0, le=1)
    sampled_frames: int = Field(ge=0)
    timeline: list[TimelinePoint]
    alert_ranges: list[AlertRange]
    key_frames: list[KeyFrame] = Field(max_length=5)


class VideoJobResponse(BaseModel):
    job_id: str
    status: str
    progress: float = Field(ge=0, le=100)
    result: VideoResult | None = None
    error: str | None = None


class LiveAlert(BaseModel):
    id: int
    timestamp: str
    source: Literal["live", "upload"]
    source_name: str
    label: str
    confidence: float = Field(ge=0, le=1)
    smoothed_confidence: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    snapshot_url: str | None = None
    video_time_seconds: float | None = Field(default=None, ge=0)


class LiveAlertSettings(BaseModel):
    threshold: float = Field(ge=0, le=1)
    sensitivity: Literal["low", "balanced", "high"] = "balanced"


class AlertStats(BaseModel):
    total_alerts: int = Field(ge=0)
    last_24h: int = Field(ge=0)
    average_confidence: float = Field(ge=0, le=1)


class AlertListResponse(BaseModel):
    alerts: list[LiveAlert]
    total: int = Field(ge=0)
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    total_pages: int = Field(ge=0)
    stats: AlertStats


class IncidentResponse(BaseModel):
    id: int
    camera_id: str
    start_time: str
    end_time: str | None = None
    peak_confidence: float = Field(ge=0, le=1)
    snapshot_url: str | None = None
    clip_url: str | None = None
    status: Literal["new", "acknowledged", "false-positive"]


class IncidentListResponse(BaseModel):
    incidents: list[IncidentResponse]
    total: int = Field(ge=0)
    page: int = Field(ge=1)
    page_size: int = Field(ge=1)
    total_pages: int = Field(ge=0)


class HealthResponse(BaseModel):
    model_loaded: bool
    device: str
    version: str


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float = Field(ge=0, le=1)
    violence_probability: float = Field(ge=0, le=1)
