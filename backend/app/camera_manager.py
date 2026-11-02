import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

from backend.app.alerts_store import AlertStore
from backend.app.config import Settings
from backend.app.incidents_store import EvidenceFrame, IncidentStore
from backend.app.model_service import ModelService
from backend.app.schemas import (
    CameraStatusResponse,
    PredictionResponse,
)
from src.infer_video import _AlertSmoother


LOGGER = logging.getLogger("mob_alert.backend")
CameraSourceType = Literal["file", "url", "webcam"]
_CAMERA_URL_SCHEMES = {"http", "https", "rtsp", "rtsps"}
_MAX_INFERENCE_BATCH_SIZE = 8


@dataclass(frozen=True)
class CameraFrame:
    image: np.ndarray
    captured_at: datetime
    captured_monotonic: float
    display_image: np.ndarray | None = None


def _normalize_source(source: str | int) -> tuple[str | int, CameraSourceType]:
    if isinstance(source, bool):
        raise ValueError("Camera source must be a URL, video file, or webcam index.")
    if isinstance(source, int):
        if source < 0:
            raise ValueError("Webcam index cannot be negative.")
        return source, "webcam"
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Camera source cannot be empty.")

    value = source.strip()
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise ValueError("Camera stream URL is malformed.") from exc
    if parsed.scheme.lower() in _CAMERA_URL_SCHEMES:
        try:
            hostname = parsed.hostname
            parsed.port
        except ValueError as exc:
            raise ValueError("Camera stream URL contains an invalid port.") from exc
        if not hostname:
            raise ValueError("Camera stream URL must include a host.")
        return value, "url"

    path = Path(value).expanduser()
    try:
        if not path.is_file():
            raise ValueError("Camera video file does not exist or is not a file.")
        return str(path.resolve()), "file"
    except OSError as exc:
        raise ValueError("Camera video file path is not accessible.") from exc


def _safe_source_display(source: str | int, source_type: CameraSourceType) -> str:
    if source_type == "webcam":
        return f"webcam:{source}"
    if source_type == "file":
        return f"file:{Path(str(source)).name}"

    parsed = urlsplit(str(source))
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    if parsed.port is not None:
        hostname = f"{hostname}:{parsed.port}"
    if parsed.username is not None or parsed.password is not None:
        hostname = f"***@{hostname}"
    return urlunsplit((parsed.scheme, hostname, "", "", ""))


def camera_worker_source(
    source_type: str,
    source_uri: str,
) -> str | int:
    if source_type == "webcam":
        if not source_uri.isdecimal():
            raise ValueError("Webcam source_uri must be a non-negative camera index.")
        return int(source_uri)
    if source_type == "rtsp":
        normalized, detected_type = _normalize_source(source_uri)
        if detected_type != "url":
            raise ValueError("RTSP source_uri must be an RTSP or HTTP stream URL.")
        return normalized
    if source_type == "file":
        normalized, detected_type = _normalize_source(source_uri)
        if detected_type != "file":
            raise ValueError("File source_uri must be a local video file path.")
        return normalized
    raise ValueError("Unsupported camera source_type.")


def capture_source_preview(
    source_type: str,
    source_uri: str,
    settings: Settings,
) -> tuple[bool, str | None, bytes | None]:
    try:
        source = camera_worker_source(source_type, source_uri)
    except ValueError as exc:
        return False, str(exc), None

    try:
        if source_type == "rtsp":
            timeout_ms = round(settings.camera_read_timeout_seconds * 1000)
            capture = cv2.VideoCapture(
                source,
                cv2.CAP_FFMPEG,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    timeout_ms,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    timeout_ms,
                ],
            )
        else:
            capture = cv2.VideoCapture(source)
    except (cv2.error, OSError, RuntimeError):
        LOGGER.exception("Could not open camera test source.")
        return False, "Could not open camera source.", None

    try:
        if not capture.isOpened():
            return False, "Could not open camera source.", None
        success, frame = capture.read()
        if not success or frame is None or not frame.size:
            return False, "Camera source opened but did not provide a frame.", None
        encoded, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 80],
        )
        if not encoded:
            return False, "Could not encode the camera preview frame.", None
        return True, None, buffer.tobytes()
    except (cv2.error, OSError, RuntimeError):
        LOGGER.exception("Could not read the camera test source.")
        return False, "Could not read a preview frame from camera source.", None
    finally:
        capture.release()


@dataclass
class _CameraRuntime:
    camera_id: str
    source: str | int
    source_type: CameraSourceType
    inference_fps: float
    worker: "CameraWorker"
    smoother: _AlertSmoother
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_inference_started: float = 0
    last_frame_inferred: float = 0
    inference_count: int = 0
    inference_rate_started: float = field(default_factory=time.monotonic)
    measured_inference_fps: float = 0
    last_inference_at: str | None = None
    inference_ms: float | None = None
    predicted_class: str | None = None
    confidence: float | None = None
    smoothed_confidence: float | None = None
    alert: bool = False
    state: Literal["NORMAL", "ALERT", "ACKNOWLEDGED", "COOLDOWN"] = "NORMAL"
    cooldown_until: float | None = None
    inference_error: str | None = None
    incident_id: int | None = None
    incident_started_monotonic: float | None = None
    incident_peak_confidence: float = 0
    incident_pre_roll: list[EvidenceFrame] = field(default_factory=list)

    def response(self) -> CameraStatusResponse:
        online, last_frame_at, input_fps, connection_error = self.worker.status()
        with self.lock:
            return CameraStatusResponse(
                camera_id=self.camera_id,
                source=_safe_source_display(self.source, self.source_type),
                source_type=self.source_type,
                online=online,
                last_frame_at=last_frame_at,
                last_inference_at=self.last_inference_at,
                inference_ms=self.inference_ms,
                predicted_class=self.predicted_class,
                confidence=self.confidence,
                smoothed_confidence=self.smoothed_confidence,
                alert=self.alert,
                state=self.state,
                input_fps=input_fps,
                inference_fps=self.measured_inference_fps,
                error=connection_error or self.inference_error,
            )


class CameraWorker:
    def __init__(
        self,
        *,
        camera_id: str,
        source: str | int,
        source_type: CameraSourceType,
        settings: Settings,
    ) -> None:
        self.camera_id = camera_id
        self._source = source
        self._source_type = source_type
        self._settings = settings
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._capture_lock = threading.Lock()
        self._capture: cv2.VideoCapture | None = None
        self._latest_frame: CameraFrame | None = None
        self._evidence_frames: deque[EvidenceFrame] = deque(
            maxlen=max(
                1,
                math.ceil(
                    settings.incident_ring_buffer_seconds
                    * settings.incident_evidence_fps
                ),
            )
        )
        self._evidence_frame_interval = 1 / settings.incident_evidence_fps
        self._last_evidence_frame_at = 0.0
        self._online = False
        self._last_frame_at: str | None = None
        self._input_fps = 0.0
        self._fps_window_started = time.monotonic()
        self._fps_window_frames = 0
        self._error: str | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"mob-alert-camera-{camera_id}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def latest_frame(self) -> CameraFrame | None:
        with self._lock:
            return self._latest_frame

    def evidence_frames(
        self,
        *,
        since_monotonic: float | None = None,
    ) -> list[EvidenceFrame]:
        with self._lock:
            return [
                (timestamp, image)
                for timestamp, image in self._evidence_frames
                if since_monotonic is None or timestamp >= since_monotonic
            ]

    def snapshot_jpeg(self) -> bytes | None:
        frame = self.latest_frame()
        if frame is None:
            return None
        image = frame.display_image if frame.display_image is not None else frame.image
        encoded, buffer = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )
        if not encoded:
            raise RuntimeError(f"Could not encode snapshot for camera {self.camera_id}.")
        return buffer.tobytes()

    def status(self) -> tuple[bool, str | None, float, str | None]:
        with self._lock:
            return self._online, self._last_frame_at, self._input_fps, self._error

    def close(self) -> None:
        self._stop.set()
        with self._capture_lock:
            capture = self._capture
        if capture is not None:
            capture.release()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def _run(self) -> None:
        backoff = min(
            self._settings.camera_reconnect_initial_seconds,
            self._settings.camera_reconnect_max_seconds,
        )
        while not self._stop.is_set():
            try:
                capture = self._open_capture()
            except (cv2.error, OSError, RuntimeError):
                LOGGER.exception("Could not open source for camera %s.", self.camera_id)
                self._set_offline("Unable to open camera source.")
                if self._stop.wait(backoff):
                    break
                backoff = min(
                    backoff * 2,
                    self._settings.camera_reconnect_max_seconds,
                )
                continue
            with self._capture_lock:
                self._capture = capture

            try:
                is_opened = capture.isOpened()
            except (cv2.error, OSError, RuntimeError):
                LOGGER.exception("Could not inspect source for camera %s.", self.camera_id)
                is_opened = False
            if not is_opened:
                capture.release()
                self._set_offline("Unable to open camera source.")
                if self._stop.wait(backoff):
                    break
                backoff = min(
                    backoff * 2,
                    self._settings.camera_reconnect_max_seconds,
                )
                continue

            try:
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except cv2.error:
                LOGGER.debug(
                    "Could not set the capture buffer size for camera %s.",
                    self.camera_id,
                    exc_info=True,
                )
            self._set_online()
            backoff = min(
                self._settings.camera_reconnect_initial_seconds,
                self._settings.camera_reconnect_max_seconds,
            )
            rewound_file = False
            file_frame_interval = self._file_frame_interval(capture)
            next_file_frame_at = time.monotonic()

            while not self._stop.is_set():
                try:
                    success, frame = capture.read()
                except (cv2.error, OSError, RuntimeError):
                    LOGGER.exception("Camera read failed for camera %s.", self.camera_id)
                    self._set_offline("Camera read failed; reconnecting.")
                    break
                if success and frame is not None and frame.size:
                    rewound_file = False
                    self._publish_frame(frame)
                    if file_frame_interval is not None:
                        next_file_frame_at += file_frame_interval
                        if self._stop.wait(
                            max(0, next_file_frame_at - time.monotonic())
                        ):
                            break
                        if time.monotonic() - next_file_frame_at > file_frame_interval:
                            next_file_frame_at = time.monotonic()
                    continue
                try:
                    loop_file = (
                        self._source_type == "file"
                        and not rewound_file
                        and capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    )
                except (cv2.error, OSError, RuntimeError):
                    LOGGER.exception(
                        "Could not rewind video for camera %s.",
                        self.camera_id,
                    )
                    loop_file = False
                if loop_file:
                    rewound_file = True
                    continue

                self._set_offline("Camera stream stopped; reconnecting.")
                break

            capture.release()
            with self._capture_lock:
                if self._capture is capture:
                    self._capture = None
            if not self._stop.wait(backoff):
                backoff = min(
                    backoff * 2,
                    self._settings.camera_reconnect_max_seconds,
                )

    def _open_capture(self) -> cv2.VideoCapture:
        if self._source_type != "url":
            return cv2.VideoCapture(self._source)

        timeout_ms = round(self._settings.camera_read_timeout_seconds * 1000)
        parameters = [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            timeout_ms,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            timeout_ms,
        ]
        return cv2.VideoCapture(self._source, cv2.CAP_FFMPEG, parameters)

    def _file_frame_interval(
        self,
        capture: cv2.VideoCapture,
    ) -> float | None:
        if self._source_type != "file":
            return None
        try:
            source_fps = capture.get(cv2.CAP_PROP_FPS)
        except cv2.error:
            LOGGER.debug(
                "Could not read the file frame rate for camera %s.",
                self.camera_id,
                exc_info=True,
            )
            source_fps = 0
        playback_fps = (
            min(source_fps, self._settings.camera_file_fps)
            if math.isfinite(source_fps) and source_fps > 0
            else self._settings.camera_file_fps
        )
        return 1 / playback_fps

    def _publish_frame(self, frame: np.ndarray) -> None:
        height, width = frame.shape[:2]
        display_scale = min(
            1.0,
            self._settings.camera_display_frame_width / width,
            self._settings.camera_display_frame_height / height,
        )
        display_frame = frame
        if display_scale < 1:
            display_frame = cv2.resize(
                frame,
                (
                    max(1, round(width * display_scale)),
                    max(1, round(height * display_scale)),
                ),
                interpolation=cv2.INTER_AREA,
            )
        scale = min(
            1.0,
            self._settings.camera_frame_width / width,
            self._settings.camera_frame_height / height,
        )
        if scale < 1:
            frame = cv2.resize(
                frame,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        captured_monotonic = time.monotonic()
        captured_at = datetime.now(timezone.utc)
        with self._lock:
            self._latest_frame = CameraFrame(
                image=frame,
                captured_at=captured_at,
                captured_monotonic=captured_monotonic,
                display_image=display_frame,
            )
            if (
                captured_monotonic - self._last_evidence_frame_at
                >= self._evidence_frame_interval
            ):
                evidence_height, evidence_width = display_frame.shape[:2]
                evidence_scale = min(
                    1.0,
                    self._settings.incident_frame_width / evidence_width,
                    self._settings.incident_frame_height / evidence_height,
                )
                evidence_image = (
                    cv2.resize(
                        display_frame,
                        (
                            max(1, round(evidence_width * evidence_scale)),
                            max(1, round(evidence_height * evidence_scale)),
                        ),
                        interpolation=cv2.INTER_AREA,
                    )
                    if evidence_scale < 1
                    else display_frame.copy()
                )
                self._evidence_frames.append((captured_monotonic, evidence_image))
                self._last_evidence_frame_at = captured_monotonic
            self._online = True
            self._last_frame_at = captured_at.isoformat()
            self._error = None
            self._fps_window_frames += 1
            elapsed = captured_monotonic - self._fps_window_started
            if elapsed >= 1:
                self._input_fps = self._fps_window_frames / elapsed
                self._fps_window_frames = 0
                self._fps_window_started = captured_monotonic

    def _set_online(self) -> None:
        with self._lock:
            self._online = True
            self._error = None

    def _set_offline(self, error: str) -> None:
        with self._lock:
            self._online = False
            self._latest_frame = None
            self._error = error


class CameraManager:
    def __init__(
        self,
        settings: Settings,
        model_service: ModelService,
        alert_store: AlertStore | None = None,
        incident_store: IncidentStore | None = None,
    ) -> None:
        if not model_service.is_loaded:
            raise RuntimeError("The classifier must be loaded before cameras start.")
        self._settings = settings
        self._model_service = model_service
        self._alert_store = alert_store
        self._incident_store = incident_store
        self._cameras: dict[str, _CameraRuntime] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._inference_tokens = 1.0
        self._last_token_update = time.monotonic()
        self._scheduler = threading.Thread(
            target=self._schedule_inference,
            name="mob-alert-inference-scheduler",
            daemon=True,
        )
        self._scheduler.start()
        self._recovery_stop = threading.Event()
        self._recovery_thread = threading.Thread(
            target=self._recover_crashed_workers,
            name="mob-alert-camera-recovery",
            daemon=True,
        )
        self._recovery_thread.start()

    def add_camera(
        self,
        camera_id: str,
        source: str | int,
        inference_fps: float | None = None,
    ) -> CameraStatusResponse:
        if not camera_id or len(camera_id) > 64:
            raise ValueError("Camera ID must be between 1 and 64 characters.")
        if not all(char.isalnum() or char in "_-" for char in camera_id):
            raise ValueError("Camera ID may contain only letters, numbers, '_' or '-'.")
        requested_fps = (
            self._settings.camera_inference_fps
            if inference_fps is None
            else inference_fps
        )
        if (
            isinstance(requested_fps, bool)
            or not math.isfinite(requested_fps)
            or requested_fps <= 0
            or requested_fps > 30
        ):
            raise ValueError("Camera inference FPS must be greater than 0 and at most 30.")

        normalized_source, source_type = _normalize_source(source)
        worker = CameraWorker(
            camera_id=camera_id,
            source=normalized_source,
            source_type=source_type,
            settings=self._settings,
        )
        runtime = _CameraRuntime(
            camera_id=camera_id,
            source=normalized_source,
            source_type=source_type,
            inference_fps=requested_fps,
            worker=worker,
            smoother=_AlertSmoother(
                self._settings.video_window_size,
                self._settings.violence_threshold,
                self._settings.video_alert_duration_seconds,
            ),
        )
        with self._lock:
            if camera_id in self._cameras:
                raise ValueError(f"Camera '{camera_id}' already exists.")
            if len(self._cameras) >= self._settings.max_cameras:
                raise RuntimeError("The configured camera limit has been reached.")
            self._cameras[camera_id] = runtime
        try:
            worker.start()
        except RuntimeError:
            with self._lock:
                self._cameras.pop(camera_id, None)
            raise
        return runtime.response()

    def list_cameras(self) -> list[CameraStatusResponse]:
        with self._lock:
            cameras = list(self._cameras.values())
        return [camera.response() for camera in cameras]

    def acknowledge(self, camera_id: str) -> CameraStatusResponse | None:
        with self._lock:
            camera = self._cameras.get(camera_id)
        if camera is None:
            return None
        with camera.lock:
            if camera.state != "ALERT":
                raise ValueError("Only active alerts can be acknowledged.")
            camera.state = "ACKNOWLEDGED"
            camera.alert = False
        return camera.response()

    def remove_camera(self, camera_id: str) -> bool:
        with self._lock:
            camera = self._cameras.pop(camera_id, None)
        if camera is None:
            return False
        self._close_active_incident(camera)
        camera.worker.close()
        return True

    def snapshot_jpeg(self, camera_id: str) -> bytes | None:
        with self._lock:
            camera = self._cameras.get(camera_id)
        return camera.worker.snapshot_jpeg() if camera is not None else None

    def latest_frame(self, camera_id: str) -> CameraFrame | None:
        with self._lock:
            camera = self._cameras.get(camera_id)
        return camera.worker.latest_frame() if camera is not None else None

    def _recover_crashed_workers(self) -> None:
        while not self._recovery_stop.wait(5):
            with self._lock:
                runtimes = list(self._cameras.values())
            for runtime in runtimes:
                worker = runtime.worker
                if getattr(worker, "_thread", None) is None:
                    continue
                if worker._thread.is_alive():
                    continue
                try:
                    new_worker = CameraWorker(
                        camera_id=runtime.camera_id,
                        source=runtime.source,
                        source_type=runtime.source_type,
                        settings=self._settings,
                    )
                    new_worker.start()
                except RuntimeError:
                    continue
                with self._lock:
                    current = self._cameras.get(runtime.camera_id)
                    if current is runtime:
                        runtime.worker = new_worker
                        LOGGER.warning("Restarted crashed camera worker %s.", runtime.camera_id)

    def close(self) -> None:
        self._stop.set()
        self._recovery_stop.set()
        if self._recovery_thread.is_alive():
            self._recovery_thread.join(timeout=2)
        if self._scheduler.is_alive():
            self._scheduler.join(timeout=2)
        with self._lock:
            cameras = list(self._cameras.values())
            self._cameras.clear()
        for camera in cameras:
            self._close_active_incident(camera)
            camera.worker.close()

    def _close_active_incident(self, camera: _CameraRuntime) -> None:
        if self._incident_store is None:
            return
        with camera.lock:
            incident_id = camera.incident_id
            incident_started = camera.incident_started_monotonic
            pre_roll = camera.incident_pre_roll
            camera.incident_id = None
            camera.incident_started_monotonic = None
            camera.incident_pre_roll = []
        if incident_id is None:
            return
        post_roll = camera.worker.evidence_frames(
            since_monotonic=incident_started,
        )
        try:
            self._incident_store.close_incident(
                incident_id,
                end_time=datetime.now(timezone.utc).isoformat(),
                frames=pre_roll + post_roll,
            )
        except (OSError, RuntimeError) as exc:
            LOGGER.exception("Could not close incident %s.", incident_id)
            with camera.lock:
                camera.inference_error = str(exc) or "Could not close camera incident."

    def _schedule_inference(self) -> None:
        token_capacity = min(
            float(self._settings.max_cameras),
            float(self._settings.camera_total_inference_fps),
        )
        while not self._stop.is_set():
            now = time.monotonic()
            elapsed = now - self._last_token_update
            self._inference_tokens = min(
                token_capacity,
                self._inference_tokens
                + elapsed * self._settings.camera_total_inference_fps,
            )
            self._last_token_update = now
            with self._lock:
                cameras = list(self._cameras.values())
            due: list[tuple[_CameraRuntime, CameraFrame, float]] = []
            for camera in cameras:
                with camera.lock:
                    interval = 1 / camera.inference_fps
                    can_infer = now - camera.last_inference_started >= interval
                    last_inference_started = camera.last_inference_started
                frame = camera.worker.latest_frame()
                if (
                    can_infer
                    and frame is not None
                    and frame.captured_monotonic > camera.last_frame_inferred
                ):
                    due.append((camera, frame, last_inference_started))

            batch_size = min(
                len(due),
                int(self._inference_tokens),
                _MAX_INFERENCE_BATCH_SIZE,
            )
            if batch_size:
                due.sort(key=lambda item: item[2])
                selected = due[:batch_size]
                self._inference_tokens -= batch_size
                self._infer_batch([(camera, frame) for camera, frame, _ in selected])
            else:
                self._stop.wait(0.01)

    def _infer_batch(
        self,
        cameras_and_frames: list[tuple[_CameraRuntime, CameraFrame]],
    ) -> None:
        started = time.monotonic()
        for camera, frame in cameras_and_frames:
            with camera.lock:
                camera.last_inference_started = started
                camera.last_frame_inferred = frame.captured_monotonic
                camera.inference_count += 1
                elapsed = started - camera.inference_rate_started
                if elapsed >= 1:
                    camera.measured_inference_fps = camera.inference_count / elapsed
                    camera.inference_count = 0
                    camera.inference_rate_started = started

        inference_started = time.perf_counter()
        try:
            predictions = self._model_service.predict_batch(
                [frame.image for _, frame in cameras_and_frames],
                wait_for_available=True,
            )
            if len(predictions) != len(cameras_and_frames):
                raise RuntimeError(
                    "The classifier returned a different number of predictions "
                    "than camera frames."
                )
        except (RuntimeError, ValueError, OSError) as exc:
            LOGGER.exception("Camera batch inference failed.")
            for camera, _ in cameras_and_frames:
                with camera.lock:
                    camera.inference_error = str(exc) or "Camera inference failed."
            return

        inference_ms = (time.perf_counter() - inference_started) * 1000
        inferred_at = datetime.now(timezone.utc).isoformat()
        for (camera, frame), prediction in zip(cameras_and_frames, predictions):
            self._update_camera_prediction(
                camera,
                frame,
                prediction,
                inference_ms,
                inferred_at,
            )

    def _update_camera_prediction(
        self,
        camera: _CameraRuntime,
        frame: CameraFrame,
        prediction: PredictionResponse,
        inference_ms: float,
        inferred_at: str,
    ) -> None:
        now = time.monotonic()
        smoothed_confidence, smoothed_alert, alert_started = camera.smoother.update(
            prediction.violence_probability,
            now,
        )
        should_record_alert = False
        should_create_incident = False
        should_close_incident = False
        incident_to_update: tuple[int, float] | None = None
        incident_to_close: tuple[int, float | None, list[EvidenceFrame]] | None = None
        with camera.lock:
            camera.last_inference_at = inferred_at
            camera.inference_ms = inference_ms
            camera.predicted_class = prediction.predicted_class
            camera.confidence = prediction.confidence
            camera.smoothed_confidence = smoothed_confidence
            if camera.state == "COOLDOWN":
                if camera.cooldown_until is not None and now >= camera.cooldown_until:
                    camera.state = "NORMAL"
                    camera.cooldown_until = None
                    camera.smoother = _AlertSmoother(
                        self._settings.video_window_size,
                        self._settings.violence_threshold,
                        self._settings.video_alert_duration_seconds,
                    )
                    smoothed_confidence, smoothed_alert, alert_started = (
                        camera.smoother.update(prediction.violence_probability, now)
                    )
                    camera.smoothed_confidence = smoothed_confidence

            if camera.state == "NORMAL" and alert_started and smoothed_alert:
                camera.state = "ALERT"
                should_record_alert = True
                should_create_incident = self._incident_store is not None
                if should_create_incident:
                    camera.incident_started_monotonic = now
                    camera.incident_peak_confidence = prediction.violence_probability
                    camera.incident_pre_roll = camera.worker.evidence_frames()
            elif camera.state in {"ALERT", "ACKNOWLEDGED"} and not smoothed_alert:
                camera.state = "COOLDOWN"
                camera.cooldown_until = (
                    now + self._settings.camera_alert_cooldown_seconds
                )
                should_close_incident = camera.incident_id is not None
                if should_close_incident:
                    incident_to_close = (
                        camera.incident_id or 0,
                        camera.incident_started_monotonic,
                        camera.incident_pre_roll,
                    )
                    camera.incident_id = None
                    camera.incident_started_monotonic = None
                    camera.incident_pre_roll = []
            if (
                camera.incident_id is not None
                and prediction.violence_probability > camera.incident_peak_confidence
            ):
                camera.incident_peak_confidence = prediction.violence_probability
                incident_to_update = (
                    camera.incident_id,
                    camera.incident_peak_confidence,
                )
            camera.alert = camera.state == "ALERT"
            camera.inference_error = None

        if should_create_incident and self._incident_store is not None:
            snapshot = frame.display_image if frame.display_image is not None else frame.image
            try:
                incident = self._incident_store.create_incident(
                    camera_id=camera.camera_id,
                    start_time=inferred_at,
                    peak_confidence=prediction.violence_probability,
                    snapshot_frame=snapshot,
                )
                with camera.lock:
                    camera.incident_id = incident.id
            except (OSError, RuntimeError) as exc:
                LOGGER.exception("Could not create incident for camera %s.", camera.camera_id)
                with camera.lock:
                    camera.inference_error = str(exc) or "Could not persist camera incident."

        if incident_to_update is not None and self._incident_store is not None:
            try:
                self._incident_store.update_peak(*incident_to_update)
            except (OSError, RuntimeError) as exc:
                LOGGER.exception(
                    "Could not update incident for camera %s.",
                    camera.camera_id,
                )
                with camera.lock:
                    camera.inference_error = str(exc) or "Could not update camera incident."

        if should_close_incident and incident_to_close is not None and self._incident_store is not None:
            incident_id, incident_started, pre_roll = incident_to_close
            post_roll = camera.worker.evidence_frames(
                since_monotonic=incident_started,
            )
            try:
                self._incident_store.close_incident(
                    incident_id,
                    end_time=inferred_at,
                    frames=pre_roll + post_roll,
                )
            except (OSError, RuntimeError) as exc:
                LOGGER.exception(
                    "Could not close incident for camera %s.",
                    camera.camera_id,
                )
                with camera.lock:
                    camera.inference_error = str(exc) or "Could not close camera incident."

        if should_record_alert and self._alert_store is not None:
            try:
                self._alert_store.record(
                    timestamp=inferred_at,
                    source="live",
                    source_name=f"Camera {camera.camera_id}",
                    label=prediction.predicted_class,
                    confidence=prediction.confidence,
                    smoothed_confidence=smoothed_confidence,
                    threshold=self._settings.violence_threshold,
                    snapshot_frame=frame.image,
                )
            except (OSError, RuntimeError) as exc:
                LOGGER.exception(
                    "Could not persist alert for camera %s.",
                    camera.camera_id,
                )
                with camera.lock:
                    camera.inference_error = str(exc) or "Could not persist camera alert."
