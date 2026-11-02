import json as _json
import logging as _logging


class _JsonFormatter(_logging.Formatter):
    def format(self, record: _logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return _json.dumps(payload)


_handler = _logging.StreamHandler()
_handler.setFormatter(_JsonFormatter())
_logging.basicConfig(level=_logging.INFO, handlers=[_handler], force=True)
LOGGER = _logging.getLogger("mob_alert.backend")

import asyncio
import base64
import json
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import AsyncGenerator, Literal

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.websockets import WebSocket, WebSocketDisconnect
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError

from backend.app.config import Settings, get_settings
from backend.app.alerts_store import AlertStore
from backend.app.auth import UserStore, decode_access_token, create_access_token
from backend.app.audit_store import AuditStore
from backend.app.camera_manager import (
    CameraManager,
    _safe_source_display,
    camera_worker_source,
    capture_source_preview,
)
from backend.app.camera_store import Camera, CameraStore
from backend.app.camera_events import CameraEvent, CameraEventHub
from backend.app.incidents_store import IncidentStatus, IncidentStore
from backend.app.live_stream import LiveSession, LiveSessionManager
from backend.app.model_service import (
    InferenceBusyError,
    InferenceTimeoutError,
    ModelService,
)

def get_current_user(request: Request):
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user


def require_roles(*roles):
    def dependency(user=Depends(get_current_user)):
        if user.role not in roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions.")
        return user
    return dependency

from backend.app.schemas import (
    AlertListResponse,
    CameraCreateRequest,
    CameraManagementListResponse,
    CameraPreviewResponse,
    CameraResponse,
    CameraListResponse,
    CameraStatusResponse,
    CameraUpdateRequest,
    HealthResponse,
    IncidentListResponse,
    IncidentResponse,
    LiveAlertSettings,
    PredictionResponse,
    VideoJobResponse,
)
from backend.app.video_jobs import VideoJobManager


APP_VERSION = "0.1.0"
APP_STARTED_AT = time.time()

LOGGER = logging.getLogger("mob_alert.backend")
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
EVENT_HEARTBEAT_SECONDS = 15
IMAGE_FORMATS = {
    "image/jpeg": "JPEG",
    "image/png": "PNG",
    "image/webp": "WEBP",
}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    model_service = ModelService(settings)
    model_service.load()
    alert_store = AlertStore(
        settings.resolved_alerts_database_path,
        settings.resolved_alert_snapshots_dir,
    )
    user_store = UserStore(settings.resolved_alerts_database_path)
    for username, password, role in (
        ("admin", "admin", "admin"),
        ("operator", "operator", "operator"),
        ("viewer", "viewer", "viewer"),
    ):
        user_store.create_user(username, password, role)  # type: ignore[arg-type]
    audit_store = AuditStore(settings.resolved_alerts_database_path)
    incident_store = IncidentStore(
        settings.resolved_alerts_database_path,
        settings.resolved_incident_evidence_dir,
        retention_days=settings.incident_retention_days,
        ring_buffer_seconds=settings.incident_ring_buffer_seconds,
        evidence_fps=settings.incident_evidence_fps,
        cleanup_interval_seconds=settings.incident_cleanup_interval_seconds,
    )
    video_job_manager = VideoJobManager(settings, model_service, alert_store)
    live_session_manager = LiveSessionManager(model_service, settings, alert_store)
    camera_manager = CameraManager(
        settings,
        model_service,
        alert_store,
        incident_store,
    )
    camera_store = CameraStore(settings.resolved_alerts_database_path)
    camera_event_hub = CameraEventHub(camera_manager)
    camera_event_task = asyncio.create_task(camera_event_hub.run())
    for camera in camera_store.list():
        if not camera.enabled or not camera.auto_start:
            continue
        try:
            camera_manager.add_camera(
                camera.id,
                camera_worker_source(camera.source_type, camera.source_uri),
            )
        except (RuntimeError, ValueError):
            LOGGER.exception("Could not start enabled camera '%s'.", camera.id)
    app.state.model_service = model_service
    app.state.video_job_manager = video_job_manager
    app.state.alert_store = alert_store
    app.state.user_store = user_store
    app.state.audit_store = audit_store
    app.state.live_session_manager = live_session_manager
    app.state.camera_manager = camera_manager
    app.state.camera_store = camera_store
    app.state.camera_event_hub = camera_event_hub
    app.state.incident_store = incident_store
    try:
        yield
    finally:
        camera_manager.close()
        camera_event_hub.close()
        camera_event_task.cancel()
        await asyncio.gather(camera_event_task, return_exceptions=True)
        camera_store.close()
        live_session_manager.close()
        video_job_manager.close()
        incident_store.close()
        alert_store.close()
        user_store.close()
        audit_store.close()
        model_service.close()
        app.state.model_service = None
        app.state.video_job_manager = None
        app.state.alert_store = None
        app.state.live_session_manager = None
        app.state.camera_manager = None
        app.state.camera_store = None
        app.state.camera_event_hub = None
        app.state.incident_store = None
    app.state.user_store = None
    app.state.audit_store = None


settings = get_settings()
app = FastAPI(
    title="Mob-Alert API",
    description="Image inference API for the Mob-Alert violence screening prototype.",
    version=APP_VERSION,
    lifespan=lifespan,
)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limit: int = 120) -> None:
        super().__init__(app)
        self.limit = limit
        self.requests: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next):
        import time as _time

        client = request.client.host if request.client else "unknown"
        now = _time.time()
        bucket = self.requests.setdefault(client, [])
        bucket[:] = [item for item in bucket if now - item < 60]
        if len(bucket) >= self.limit:
            from fastapi.responses import JSONResponse

            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded."})
        bucket.append(now)
        return await call_next(request)


class AuthMiddleware(BaseHTTPMiddleware):
    PUBLIC_PATHS = {"/health", "/auth/login", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in self.PUBLIC_PATHS or path.startswith("/docs") or path.startswith("/openapi"):
            return await call_next(request)
        if path.startswith("/ws/"):
            token = request.query_params.get("token") or request.headers.get("authorization", "").replace("Bearer ", "")
        else:
            auth_header = request.headers.get("authorization", "")
            token = request.query_params.get("token")
            if auth_header.lower().startswith("bearer "):
                token = auth_header.split(" ", 1)[1].strip()
        if not token:
            from fastapi.responses import JSONResponse

            return JSONResponse(status_code=401, content={"detail": "Authentication required."})
        try:
            user = decode_access_token(token, get_settings().jwt_secret)
        except ValueError:
            from fastapi.responses import JSONResponse

            return JSONResponse(status_code=401, content={"detail": "Invalid or expired token."})
        request.state.user = user
        return await call_next(request)


app.add_middleware(RateLimitMiddleware, limit=settings.rate_limit_per_minute)
app.add_middleware(AuthMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_model_service(request: Request) -> ModelService:
    model_service = getattr(request.app.state, "model_service", None)
    if not isinstance(model_service, ModelService) or not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="The classifier is not loaded.")
    return model_service


def get_video_job_manager(request: Request) -> VideoJobManager:
    manager = getattr(request.app.state, "video_job_manager", None)
    if not isinstance(manager, VideoJobManager):
        raise HTTPException(status_code=503, detail="Video processing is unavailable.")
    return manager


def get_live_session_manager(request: Request) -> LiveSessionManager:
    manager = getattr(request.app.state, "live_session_manager", None)
    if not isinstance(manager, LiveSessionManager):
        raise HTTPException(status_code=503, detail="Live inference is unavailable.")
    return manager


def get_alert_store(request: Request) -> AlertStore:
    alert_store = getattr(request.app.state, "alert_store", None)
    if not isinstance(alert_store, AlertStore):
        raise HTTPException(status_code=503, detail="The alert store is unavailable.")
    return alert_store


def get_camera_manager(request: Request) -> CameraManager:
    manager = getattr(request.app.state, "camera_manager", None)
    if not isinstance(manager, CameraManager):
        raise HTTPException(status_code=503, detail="Camera monitoring is unavailable.")
    return manager


def get_camera_store(request: Request) -> CameraStore:
    store = getattr(request.app.state, "camera_store", None)
    if not isinstance(store, CameraStore):
        raise HTTPException(status_code=503, detail="Camera management is unavailable.")
    return store


def get_camera_event_hub(request: Request) -> CameraEventHub:
    hub = getattr(request.app.state, "camera_event_hub", None)
    if not isinstance(hub, CameraEventHub):
        raise HTTPException(status_code=503, detail="Camera events are unavailable.")
    return hub


def get_incident_store(request: Request) -> IncidentStore:
    store = getattr(request.app.state, "incident_store", None)
    if not isinstance(store, IncidentStore):
        raise HTTPException(status_code=503, detail="Incident storage is unavailable.")
    return store

def get_user_store(request: Request) -> UserStore:
    store = getattr(request.app.state, "user_store", None)
    if not isinstance(store, UserStore):
        raise HTTPException(status_code=503, detail="User storage is unavailable.")
    return store


def get_audit_store(request: Request) -> AuditStore:
    store = getattr(request.app.state, "audit_store", None)
    if not isinstance(store, AuditStore):
        raise HTTPException(status_code=503, detail="Audit storage is unavailable.")
    return store


def _incident_time_range_is_reversed(
    start_time: datetime,
    end_time: datetime,
) -> bool:
    normalized_start = (
        start_time.replace(tzinfo=timezone.utc)
        if start_time.tzinfo is None
        else start_time.astimezone(timezone.utc)
    )
    normalized_end = (
        end_time.replace(tzinfo=timezone.utc)
        if end_time.tzinfo is None
        else end_time.astimezone(timezone.utc)
    )
    return normalized_start > normalized_end


@app.get("/stream-test", include_in_schema=False)
def stream_test_page() -> FileResponse:
    page_path = Path(__file__).resolve().parents[1] / "stream_wall_test.html"
    if not page_path.is_file():
        raise HTTPException(status_code=404, detail="Stream test page was not found.")
    return FileResponse(page_path, media_type="text/html")


def _camera_response(
    camera: Camera,
    manager: CameraManager,
) -> CameraResponse:
    status = next(
        (
            item
            for item in manager.list_cameras()
            if item.camera_id == camera.id
        ),
        None,
    )
    if status is None:
        source_type = camera.source_type
        source_display = (
            camera.source_uri
            if source_type == "file"
            else f"webcam:{camera.source_uri}"
            if source_type == "webcam"
            else camera.source_uri
        )
        if source_type == "rtsp":
            source_display = _safe_source_display(camera.source_uri, "url")
        status = CameraStatusResponse(
            camera_id=camera.id,
            source=source_display,
            source_type=source_type,
            online=False,
        )

    safe_uri = (
        _safe_source_display(camera.source_uri, "url")
        if camera.source_type == "rtsp"
        else camera.source_uri
    )
    response_data = status.model_dump()
    response_data.update(
        {
            "source_type": camera.source_type,
            "id": camera.id,
            "name": camera.name,
            "location": camera.location,
            "source_uri": safe_uri,
            "enabled": camera.enabled,
            "created_at": camera.created_at.isoformat(),
        }
    )
    return CameraResponse(
        **response_data,
    )


def _normalize_camera_request(
    camera: CameraCreateRequest | CameraUpdateRequest,
    *,
    existing: Camera | None = None,
) -> tuple[str, str, str, str, bool, bool]:
    name = camera.name if camera.name is not None else (existing.name if existing else None)
    location = (
        camera.location
        if camera.location is not None
        else (existing.location if existing else "")
    )
    source_type = (
        camera.source_type
        if camera.source_type is not None
        else (existing.source_type if existing else None)
    )
    source_uri = (
        camera.source_uri
        if camera.source_uri is not None
        else (existing.source_uri if existing else None)
    )
    enabled = (
        camera.enabled
        if camera.enabled is not None
        else (existing.enabled if existing else True)
    )
    if camera.auto_start is None:
        auto_start = source_type != "webcam" if source_type is not None else (existing.auto_start if existing else True)
    else:
        auto_start = bool(camera.auto_start)
    if existing is not None and camera.auto_start is None:
        auto_start = existing.auto_start
    if not name or source_type is None or source_uri is None:
        raise ValueError("Camera name, source_type, and source_uri are required.")

    normalized_source = camera_worker_source(source_type, source_uri)
    return name, location, source_type, str(normalized_source), enabled, auto_start


@app.get("/cameras", response_model=CameraManagementListResponse)
def list_cameras(
    manager: CameraManager = Depends(get_camera_manager),
    store: CameraStore = Depends(get_camera_store),
) -> CameraManagementListResponse:
    return CameraManagementListResponse(
        cameras=[_camera_response(camera, manager) for camera in store.list()]
    )


@app.get("/cameras/status", response_model=CameraListResponse)
def get_camera_status(
    manager: CameraManager = Depends(get_camera_manager),
) -> CameraListResponse:
    return CameraListResponse(cameras=manager.list_cameras())


@app.get("/stream/{camera_id}")
def stream_camera(
    camera_id: str,
    quality: Literal["grid", "full"] = Query(default="grid"),
    manager: CameraManager = Depends(get_camera_manager),
    store: CameraStore = Depends(get_camera_store),
) -> StreamingResponse:
    camera = store.get(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera was not found.")
    if not camera.enabled:
        raise HTTPException(status_code=409, detail="Camera is disabled.")

    return StreamingResponse(
        _mjpeg_stream(
            manager,
            camera_id,
            quality=quality,
        ),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


async def _mjpeg_stream(
    manager: CameraManager,
    camera_id: str,
    *,
    quality: Literal["grid", "full"],
):
    max_width = 480 if quality == "grid" else 960
    frame_interval = 1 / 15
    next_frame_at = time.monotonic()
    while True:
        frame = manager.latest_frame(camera_id)
        if frame is None:
            await asyncio.sleep(0.1)
            continue

        image = frame.display_image if frame.display_image is not None else frame.image
        height, width = image.shape[:2]
        if width > max_width:
            scaled_height = max(1, round(height * max_width / width))
            image = cv2.resize(
                image,
                (max_width, scaled_height),
                interpolation=cv2.INTER_AREA,
            )
        encoded, buffer = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 78 if quality == "grid" else 88],
        )
        if encoded:
            payload = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: "
                + str(len(payload)).encode("ascii")
                + b"\r\n\r\n"
                + payload
                + b"\r\n"
            )
        next_frame_at += frame_interval
        await asyncio.sleep(max(0, next_frame_at - time.monotonic()))
        if time.monotonic() - next_frame_at > frame_interval:
            next_frame_at = time.monotonic()


@app.websocket("/ws/events")
async def camera_events_websocket(websocket: WebSocket) -> None:
    token = websocket.query_params.get("token")
    auth_header = websocket.headers.get("authorization", "")
    if not token and auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
    if not token:
        await websocket.close(code=4401, reason="Authentication required.")
        return
    try:
        websocket.app.state.current_user = decode_access_token(token, get_settings().jwt_secret)
    except ValueError:
        await websocket.close(code=4401, reason="Invalid or expired token.")
        return
    hub = getattr(websocket.app.state, "camera_event_hub", None)
    if not isinstance(hub, CameraEventHub):
        await websocket.close(code=1013, reason="Camera events are unavailable.")
        return
    await websocket.accept()
    queue = hub.subscribe()
    sender = asyncio.create_task(_send_camera_events(websocket, queue))
    receiver = asyncio.create_task(_receive_camera_events(websocket))
    try:
        done, pending = await asyncio.wait(
            {sender, receiver},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in done:
            if task.cancelled():
                continue
            exception = task.exception()
            if exception is not None:
                LOGGER.debug("Camera event WebSocket closed: %s", exception)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    finally:
        hub.unsubscribe(queue)
        for task in (sender, receiver):
            if not task.done():
                task.cancel()
        await asyncio.gather(sender, receiver, return_exceptions=True)


async def _send_camera_events(
    websocket: WebSocket,
    queue: asyncio.Queue[CameraEvent],
) -> None:
    next_heartbeat_at = time.monotonic() + EVENT_HEARTBEAT_SECONDS
    while True:
        try:
            timeout = max(0, next_heartbeat_at - time.monotonic())
            event = await asyncio.wait_for(queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            await websocket.send_json(
                {
                    "type": "heartbeat",
                    "timestamp": time.time(),
                }
            )
            next_heartbeat_at = time.monotonic() + EVENT_HEARTBEAT_SECONDS
            continue
        await websocket.send_json(event)
        if time.monotonic() >= next_heartbeat_at:
            await websocket.send_json(
                {
                    "type": "heartbeat",
                    "timestamp": time.time(),
                }
            )
            next_heartbeat_at = time.monotonic() + EVENT_HEARTBEAT_SECONDS


async def _receive_camera_events(websocket: WebSocket) -> None:
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return


@app.get("/cameras/{camera_id}", response_model=CameraResponse)
def get_camera(
    camera_id: str,
    manager: CameraManager = Depends(get_camera_manager),
    store: CameraStore = Depends(get_camera_store),
) -> CameraResponse:
    camera = store.get(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera was not found.")
    return _camera_response(camera, manager)


@app.post("/cameras/{camera_id}/acknowledge", response_model=CameraStatusResponse)
def acknowledge_camera_alert(
    camera_id: str,
    manager: CameraManager = Depends(get_camera_manager),
    user=Depends(require_roles("admin", "operator")),
) -> CameraStatusResponse:
    try:
        status = manager.acknowledge(camera_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if status is None:
        raise HTTPException(status_code=404, detail="Camera was not found.")
    return status


@app.post(
    "/cameras",
    response_model=CameraResponse,
    status_code=201,
)
def add_camera(
    camera: CameraCreateRequest,
    manager: CameraManager = Depends(get_camera_manager),
    store: CameraStore = Depends(get_camera_store),
    user=Depends(require_roles("admin")),
) -> CameraResponse:
    try:
        name, location, source_type, source_uri, enabled, auto_start = _normalize_camera_request(
            camera
        )
        camera_id = camera.camera_id or camera.id
        if camera.camera_id is not None:
            camera_id = camera.camera_id
        if len(camera_id) > 64 or not all(
            char.isalnum() or char in "_-" for char in camera_id
        ):
            raise ValueError("Camera ID may contain only letters, numbers, '_' or '-'.")
        camera.id = camera_id
        if store.get(camera_id) is not None:
            raise HTTPException(status_code=409, detail="Camera ID already exists.")
        record = store.create(
            Camera(
                id=camera_id,
                name=name,
                location=location,
                source_type=source_type,
                source_uri=source_uri,
                enabled=enabled,
                auto_start=auto_start,
            )
        )
        try:
            if enabled:
                manager.add_camera(
                    camera_id,
                    camera_worker_source(source_type, source_uri),
                    camera.inference_fps,
                )
        except ValueError as exc:
            store.delete(camera_id)
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            store.delete(camera_id)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return _camera_response(record, manager)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.put("/cameras/{camera_id}", response_model=CameraResponse)
def update_camera(
    camera_id: str,
    changes: CameraUpdateRequest,
    manager: CameraManager = Depends(get_camera_manager),
    store: CameraStore = Depends(get_camera_store),
    user=Depends(require_roles("admin")),
) -> CameraResponse:
    existing = store.get(camera_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Camera was not found.")
    try:
        name, location, source_type, source_uri, enabled, auto_start = _normalize_camera_request(
            changes,
            existing=existing,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    updated = store.update(
        camera_id,
        name=name,
        location=location,
        source_type=source_type,
        source_uri=source_uri,
        enabled=enabled,
        auto_start=auto_start,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Camera was not found.")

    source_changed = (
        existing.source_type != source_type or existing.source_uri != source_uri
    )
    try:
        if existing.enabled and (not enabled or source_changed):
            manager.remove_camera(camera_id)
        if enabled and (not existing.enabled or source_changed):
            manager.add_camera(
                camera_id,
                camera_worker_source(source_type, source_uri),
            )
    except (ValueError, RuntimeError) as exc:
        store.update(
            camera_id,
            name=existing.name,
            location=existing.location,
            source_type=existing.source_type,
            source_uri=existing.source_uri,
            enabled=existing.enabled,
        )
        if existing.enabled:
            try:
                manager.add_camera(
                    camera_id,
                    camera_worker_source(existing.source_type, existing.source_uri),
                )
            except (ValueError, RuntimeError):
                LOGGER.exception("Could not restore camera '%s' after update failure.", camera_id)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _camera_response(updated, manager)


@app.delete("/cameras/{camera_id}", status_code=204)
def remove_camera(
    camera_id: str,
    manager: CameraManager = Depends(get_camera_manager),
    store: CameraStore = Depends(get_camera_store),
    user=Depends(require_roles("admin")),
) -> Response:
    if not store.delete(camera_id):
        raise HTTPException(status_code=404, detail="Camera was not found.")
    manager.remove_camera(camera_id)
    return Response(status_code=204)


@app.post(
    "/cameras/{camera_id}/test",
    response_model=CameraPreviewResponse,
)
def test_camera(
    camera_id: str,
    store: CameraStore = Depends(get_camera_store),
    runtime_settings: Settings = Depends(get_settings),
) -> CameraPreviewResponse:
    camera = store.get(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera was not found.")
    ok, error, preview_frame = capture_source_preview(
        camera.source_type,
        camera.source_uri,
        runtime_settings,
    )
    return CameraPreviewResponse(
        ok=ok,
        error=error,
        preview_frame_base64=(
            base64.b64encode(preview_frame).decode("ascii")
            if preview_frame is not None
            else None
        ),
        preview_mime_type="image/jpeg" if preview_frame is not None else None,
    )


@app.get("/cameras/{camera_id}/snapshot")
def get_camera_snapshot(
    camera_id: str,
    manager: CameraManager = Depends(get_camera_manager),
    store: CameraStore = Depends(get_camera_store),
) -> Response:
    camera = store.get(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera was not found.")
    if not camera.enabled:
        raise HTTPException(status_code=409, detail="Camera is disabled.")
    try:
        snapshot = manager.snapshot_jpeg(camera_id)
    except RuntimeError as exc:
        LOGGER.exception("Could not encode camera snapshot for '%s'.", camera_id)
        raise HTTPException(status_code=500, detail="Could not encode camera snapshot.") from exc
    if snapshot is None:
        raise HTTPException(status_code=503, detail="Camera has no current frame.")
    return Response(content=snapshot, media_type="image/jpeg")


@app.post("/cameras/{camera_id}/start", response_model=CameraResponse)
def start_camera(
    camera_id: str,
    manager: CameraManager = Depends(get_camera_manager),
    store: CameraStore = Depends(get_camera_store),
    user=Depends(require_roles("admin", "operator")),
) -> CameraResponse:
    camera = store.get(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera was not found.")
    if not camera.enabled:
        raise HTTPException(status_code=409, detail="Camera is disabled.")
    try:
        manager.add_camera(
            camera_id,
            camera_worker_source(camera.source_type, camera.source_uri),
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _camera_response(camera, manager)


@app.post("/cameras/{camera_id}/stop", response_model=CameraResponse)
def stop_camera(
    camera_id: str,
    manager: CameraManager = Depends(get_camera_manager),
    store: CameraStore = Depends(get_camera_store),
    user=Depends(require_roles("admin", "operator")),
) -> CameraResponse:
    camera = store.get(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="Camera was not found.")
    manager.remove_camera(camera_id)
    return _camera_response(camera, manager)



@app.post("/auth/login")
def login(payload: dict, user_store: UserStore = Depends(get_user_store)) -> dict:
    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))
    user = user_store.authenticate(username, password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    settings = get_settings()
    token = create_access_token(user, settings.jwt_secret, settings.jwt_expire_minutes)
    return {"access_token": token, "token_type": "bearer", "role": user.role, "username": user.username}


@app.get("/auth/me")
def me(user=Depends(get_current_user)) -> dict:
    return {"username": user.username, "role": user.role}


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    model_service = getattr(request.app.state, "model_service", None)
    is_loaded = isinstance(model_service, ModelService) and model_service.is_loaded
    device = model_service.device_name if is_loaded else "unavailable"
    return HealthResponse(
        model_loaded=is_loaded,
        device=device,
        version=APP_VERSION,
    )

@app.get("/health/system")
def system_health(
    manager: CameraManager = Depends(get_camera_manager),
    user=Depends(get_current_user),
) -> dict:
    uptime_seconds = time.time() - APP_STARTED_AT
    gpu_memory = None
    try:
        import torch

        if torch.cuda.is_available():
            gpu_memory = {
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
            }
    except Exception:  # noqa: BLE001
        gpu_memory = None
    cameras = []
    try:
        for status in manager.list_cameras():
            cameras.append(
                {
                    "camera_id": status.camera_id,
                    "online": status.online,
                    "input_fps": status.input_fps,
                    "inference_fps": status.inference_fps,
                    "inference_ms": status.inference_ms,
                    "state": status.state,
                    "error": status.error,
                }
            )
    except Exception:  # noqa: BLE001
        cameras = []
    return {
        "uptime_seconds": uptime_seconds,
        "gpu_memory": gpu_memory,
        "cameras": cameras,
    }


@app.post("/predict-image", response_model=PredictionResponse)
def predict_image(
    file: UploadFile = File(...),
    model_service: ModelService = Depends(get_model_service),
    runtime_settings: Settings = Depends(get_settings),
) -> PredictionResponse:
    content_type = (file.content_type or "").split(";", maxsplit=1)[0].strip().lower()
    try:
        if content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=415,
                detail="Unsupported image type. Upload a JPEG, PNG, or WebP image.",
            )

        contents = file.file.read(runtime_settings.max_upload_size_bytes + 1)
        if len(contents) > runtime_settings.max_upload_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Image exceeds the maximum upload size of "
                    f"{runtime_settings.max_upload_size_mb} MB."
                ),
            )
        if not contents:
            raise HTTPException(status_code=400, detail="The uploaded image is empty.")

        try:
            with Image.open(BytesIO(contents)) as uploaded_image:
                if uploaded_image.format != IMAGE_FORMATS[content_type]:
                    raise HTTPException(
                        status_code=400,
                        detail="The image contents do not match the declared file type.",
                    )
                width, height = uploaded_image.size
                if width * height > runtime_settings.max_image_pixels:
                    raise HTTPException(
                        status_code=413,
                        detail="Image dimensions exceed the configured pixel limit.",
                    )
                image = cv2.cvtColor(
                    np.asarray(uploaded_image.convert("RGB")),
                    cv2.COLOR_RGB2BGR,
                )
        except Image.DecompressionBombError as exc:
            raise HTTPException(
                status_code=413,
                detail="Image dimensions exceed the configured pixel limit.",
            ) from exc
        except (OSError, UnidentifiedImageError) as exc:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is not a valid JPEG, PNG, or WebP image.",
            ) from exc

        return model_service.predict(image)
    except InferenceBusyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except InferenceTimeoutError as exc:
        LOGGER.warning("Image prediction timed out: %s", exc)
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except (RuntimeError, ValueError, OSError) as exc:
        LOGGER.exception("Image prediction failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Image prediction failed: {exc}",
        ) from exc
    finally:
        file.file.close()


@app.post("/predict-video", response_model=VideoJobResponse, status_code=202)
async def predict_video(
    file: UploadFile = File(...),
    manager: VideoJobManager = Depends(get_video_job_manager),
    runtime_settings: Settings = Depends(get_settings),
) -> VideoJobResponse:
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        await file.close()
        raise HTTPException(
            status_code=415,
            detail="Unsupported video type. Upload an MP4, AVI, or MOV file.",
        )

    temporary_path: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="mob-alert-",
            suffix=suffix,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        size = 0
        with temporary_path.open("wb") as temporary_file:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > runtime_settings.max_video_upload_size_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Video exceeds the maximum upload size of "
                            f"{runtime_settings.max_video_upload_size_mb} MB."
                        ),
                    )
                temporary_file.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="The uploaded video is empty.")

        try:
            return manager.submit(temporary_path, source_name=filename)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    except HTTPException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    except OSError as exc:
        LOGGER.exception("Could not save uploaded video.")
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Could not save uploaded video: {exc}",
        ) from exc
    finally:
        await file.close()


@app.get("/jobs/{job_id}", response_model=VideoJobResponse)
def get_video_job(
    job_id: str,
    manager: VideoJobManager = Depends(get_video_job_manager),
) -> VideoJobResponse:
    job = manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Video job was not found.")
    return job


@app.websocket("/ws/live")
async def live_websocket(websocket: WebSocket) -> None:
    token = websocket.query_params.get("token")
    auth_header = websocket.headers.get("authorization", "")
    if not token and auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
    if not token:
        await websocket.close(code=4401, reason="Authentication required.")
        return
    try:
        decode_access_token(token, get_settings().jwt_secret)
    except ValueError:
        await websocket.close(code=4401, reason="Invalid or expired token.")
        return
    await websocket.accept()
    manager = getattr(websocket.app.state, "live_session_manager", None)
    if not isinstance(manager, LiveSessionManager):
        await websocket.close(code=1013, reason="Live inference is unavailable.")
        return
    try:
        session = manager.open_session(asyncio.get_running_loop())
    except RuntimeError as exc:
        await websocket.close(code=1013, reason=str(exc))
        return

    sender = asyncio.create_task(_send_live_results(websocket, session))
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            payload = message.get("bytes")
            text_payload = message.get("text")
            if text_payload is not None:
                if len(text_payload) > 1024:
                    session.report_error(
                        "Live alert settings message is too large."
                    )
                    continue
                try:
                    decoded_settings = json.loads(text_payload)
                    alert_settings = LiveAlertSettings.model_validate(
                        decoded_settings
                    )
                except (json.JSONDecodeError, ValidationError, TypeError) as exc:
                    LOGGER.warning("Rejected invalid live alert settings: %s", exc)
                    session.report_error(
                        "Live settings must include a threshold from 0 to 1 "
                        "and sensitivity low, balanced, or high."
                    )
                    continue
                session.configure_alerts(
                    alert_settings.threshold,
                    alert_settings.sensitivity,
                )
                continue
            if payload is None:
                await websocket.close(
                    code=1003,
                    reason="Send JPEG frames as binary WebSocket messages.",
                )
                break
            if len(payload) > manager.max_frame_size_bytes:
                await websocket.close(
                    code=1009,
                    reason="Live frame exceeds the configured size limit.",
                )
                break
            if not payload:
                await websocket.send_json({"error": "Live frame cannot be empty."})
                continue
            session.submit_frame(payload)
    except WebSocketDisconnect:
        pass
    finally:
        manager.close_session(session)
        sender.cancel()
        await asyncio.gather(sender, return_exceptions=True)


async def _send_live_results(websocket: WebSocket, session: LiveSession) -> None:
    while True:
        result = await session.next_result()
        await websocket.send_json(result)


@app.get("/alerts", response_model=AlertListResponse)
def get_alerts(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    source: Literal["live", "upload"] | None = None,
    search: str | None = Query(default=None, min_length=1, max_length=200),
    alert_store: AlertStore = Depends(get_alert_store),
) -> AlertListResponse:
    try:
        return alert_store.list_alerts(
            page=page,
            page_size=page_size,
            source=source,
            search=search,
        )
    except (OSError, RuntimeError, SQLAlchemyError) as exc:
        LOGGER.exception("Could not query persisted alerts.")
        raise HTTPException(
            status_code=500,
            detail=f"Could not load alerts: {exc}",
        ) from exc


@app.get("/incidents", response_model=IncidentListResponse)
def get_incidents(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    camera_id: str | None = Query(default=None, min_length=1, max_length=64),
    status: IncidentStatus | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    search: str | None = Query(default=None, min_length=1, max_length=200),
    incident_store: IncidentStore = Depends(get_incident_store),
) -> IncidentListResponse:
    if (
        start_time is not None
        and end_time is not None
        and _incident_time_range_is_reversed(start_time, end_time)
    ):
        raise HTTPException(
            status_code=422,
            detail="start_time must be earlier than or equal to end_time.",
        )
    try:
        return incident_store.list_incidents(
            page=page,
            page_size=page_size,
            camera_id=camera_id,
            status=status,
            start_time=start_time,
            end_time=end_time,
            search=search,
        )
    except (OSError, RuntimeError, SQLAlchemyError) as exc:
        LOGGER.exception("Could not query persisted incidents.")
        raise HTTPException(
            status_code=500,
            detail=f"Could not load incidents: {exc}",
        ) from exc


@app.get("/incidents/export.csv")
def download_incidents_csv(
    camera_id: str | None = Query(default=None, min_length=1, max_length=64),
    status: IncidentStatus | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    search: str | None = Query(default=None, min_length=1, max_length=200),
    incident_store: IncidentStore = Depends(get_incident_store),
) -> Response:
    if (
        start_time is not None
        and end_time is not None
        and _incident_time_range_is_reversed(start_time, end_time)
    ):
        raise HTTPException(
            status_code=422,
            detail="start_time must be earlier than or equal to end_time.",
        )
    try:
        content = incident_store.csv_content(
            camera_id=camera_id,
            status=status,
            start_time=start_time,
            end_time=end_time,
            search=search,
        )
    except (OSError, RuntimeError, SQLAlchemyError) as exc:
        LOGGER.exception("Could not export persisted incidents.")
        raise HTTPException(
            status_code=500,
            detail=f"Could not export incidents: {exc}",
        ) from exc
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="incidents.csv"'},
    )


@app.get("/incidents/{incident_id}/snapshot")
def get_incident_snapshot(
    incident_id: int,
    incident_store: IncidentStore = Depends(get_incident_store),
) -> FileResponse:
    try:
        path = incident_store.get_snapshot_path(incident_id)
    except (OSError, RuntimeError, SQLAlchemyError) as exc:
        LOGGER.exception("Could not load incident snapshot %s.", incident_id)
        raise HTTPException(
            status_code=500,
            detail=f"Could not load incident snapshot: {exc}",
        ) from exc
    if path is None:
        raise HTTPException(status_code=404, detail="Incident snapshot was not found.")
    return FileResponse(
        path,
        media_type="image/jpeg",
        filename=f"incident-{incident_id}.jpg",
        content_disposition_type="inline",
    )


@app.get("/incidents/{incident_id}/clip")
def get_incident_clip(
    incident_id: int,
    incident_store: IncidentStore = Depends(get_incident_store),
) -> FileResponse:
    try:
        path = incident_store.get_clip_path(incident_id)
    except (OSError, RuntimeError, SQLAlchemyError) as exc:
        LOGGER.exception("Could not load incident clip %s.", incident_id)
        raise HTTPException(
            status_code=500,
            detail=f"Could not load incident clip: {exc}",
        ) from exc
    if path is None:
        raise HTTPException(status_code=404, detail="Incident clip was not found.")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"incident-{incident_id}.mp4",
        content_disposition_type="inline",
    )


@app.post(
    "/incidents/{incident_id}/acknowledge",
    response_model=IncidentResponse,
)
def acknowledge_incident(
    incident_id: int,
    incident_store: IncidentStore = Depends(get_incident_store),
    audit_store: AuditStore = Depends(get_audit_store),
    user=Depends(require_roles("admin", "operator")),
) -> IncidentResponse:
    try:
        incident = incident_store.acknowledge(incident_id)
    except (OSError, RuntimeError, SQLAlchemyError) as exc:
        LOGGER.exception("Could not acknowledge incident %s.", incident_id)
        raise HTTPException(
            status_code=500,
            detail=f"Could not acknowledge incident: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident was not found.")
    audit_store.record(username=user.username, role=user.role, action="acknowledge_incident", target=str(incident_id))

    return incident


@app.post(
    "/incidents/{incident_id}/false-positive",
    response_model=IncidentResponse,
)
def mark_incident_false_positive(
    incident_id: int,
    incident_store: IncidentStore = Depends(get_incident_store),
    audit_store: AuditStore = Depends(get_audit_store),
    user=Depends(require_roles("admin", "operator")),
) -> IncidentResponse:
    try:
        incident = incident_store.mark_false_positive(incident_id)
    except (OSError, RuntimeError, SQLAlchemyError) as exc:
        LOGGER.exception("Could not mark incident %s as false-positive.", incident_id)
        raise HTTPException(
            status_code=500,
            detail=f"Could not update incident: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident was not found.")
    audit_store.record(username=user.username, role=user.role, action="mark_false_positive", target=str(incident_id))

    return incident


@app.get("/incidents/{incident_id}", response_model=IncidentResponse)
def get_incident(
    incident_id: int,
    incident_store: IncidentStore = Depends(get_incident_store),
) -> IncidentResponse:
    try:
        incident = incident_store.get_incident(incident_id)
    except (OSError, RuntimeError, SQLAlchemyError) as exc:
        LOGGER.exception("Could not load incident %s.", incident_id)
        raise HTTPException(
            status_code=500,
            detail=f"Could not load incident: {exc}",
        ) from exc
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident was not found.")
    return incident


@app.get("/alerts.csv")
def download_alerts_csv(
    source: Literal["live", "upload"] | None = None,
    search: str | None = Query(default=None, min_length=1, max_length=200),
    alert_store: AlertStore = Depends(get_alert_store),
) -> Response:
    try:
        content = alert_store.csv_content(source=source, search=search)
    except (OSError, RuntimeError, SQLAlchemyError) as exc:
        LOGGER.exception("Could not export persisted alerts.")
        raise HTTPException(
            status_code=500,
            detail=f"Could not export alerts: {exc}",
        ) from exc
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="alerts.csv"'},
    )


@app.get("/alerts/{alert_id}/snapshot")
def get_alert_snapshot(
    alert_id: int,
    alert_store: AlertStore = Depends(get_alert_store),
) -> FileResponse:
    try:
        snapshot_path = alert_store.get_snapshot_path(alert_id)
    except SQLAlchemyError as exc:
        LOGGER.exception("Could not load snapshot metadata for alert %s.", alert_id)
        raise HTTPException(
            status_code=500,
            detail=f"Could not load alert snapshot: {exc}",
        ) from exc
    if snapshot_path is None:
        raise HTTPException(status_code=404, detail="Alert snapshot was not found.")
    return FileResponse(
        snapshot_path,
        media_type="image/jpeg",
        filename=f"alert-{alert_id}.jpg",
        content_disposition_type="inline",
    )
