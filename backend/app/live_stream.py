import asyncio
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from io import BytesIO
from typing import Any

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from backend.app.alerts_store import AlertStore
from backend.app.config import Settings
from backend.app.model_service import ModelService
from src.infer_video import _AlertSmoother


LOGGER = logging.getLogger("mob_alert.backend")


def decode_jpeg_frame(payload: bytes, max_pixels: int) -> np.ndarray:
    try:
        with Image.open(BytesIO(payload)) as uploaded_image:
            if uploaded_image.format != "JPEG":
                raise ValueError("Live frames must be JPEG images.")
            width, height = uploaded_image.size
            if width <= 0 or height <= 0 or width * height > max_pixels:
                raise ValueError("Live frame dimensions exceed the configured pixel limit.")
            image = cv2.cvtColor(
                np.asarray(uploaded_image.convert("RGB")),
                cv2.COLOR_RGB2BGR,
            )
    except Image.DecompressionBombError as exc:
        raise ValueError("Live frame dimensions exceed the configured pixel limit.") from exc
    except (OSError, UnidentifiedImageError, UnicodeError) as exc:
        raise ValueError("Live frame is not a valid JPEG image.") from exc
    return image


class LatestFrameSlot:
    def __init__(self) -> None:
        self._frames: queue.Queue[bytes] = queue.Queue(maxsize=1)
        self.dropped_frames = 0

    def put(self, payload: bytes) -> None:
        try:
            self._frames.put_nowait(payload)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            else:
                self.dropped_frames += 1
            self._frames.put_nowait(payload)

    def get(self, timeout: float) -> bytes | None:
        try:
            return self._frames.get(timeout=timeout)
        except queue.Empty:
            return None


class LiveSession:
    def __init__(
        self,
        *,
        model_service: ModelService,
        settings: Settings,
        alert_store: AlertStore,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._model_service = model_service
        self._settings = settings
        self._alert_store = alert_store
        self._loop = loop
        self._frames = LatestFrameSlot()
        self._outgoing: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1)
        self._stop = threading.Event()
        self._alert_settings_lock = threading.Lock()
        self._smoother = _AlertSmoother(
            settings.video_window_size,
            settings.violence_threshold,
            settings.video_alert_duration_seconds,
        )
        self._started_at = time.monotonic()
        self._worker = threading.Thread(
            target=self._process_frames,
            name="mob-alert-live",
            daemon=True,
        )

    def start(self) -> None:
        self._worker.start()

    def submit_frame(self, payload: bytes) -> None:
        self._frames.put(payload)

    def report_error(self, message: str) -> None:
        self._publish({"error": message})

    def configure_alerts(self, threshold: float, sensitivity: str) -> None:
        duration_seconds = {
            "low": 4.0,
            "balanced": self._settings.video_alert_duration_seconds,
            "high": 0.5,
        }[sensitivity]
        with self._alert_settings_lock:
            self._smoother = _AlertSmoother(
                self._settings.video_window_size,
                threshold,
                duration_seconds,
            )

    async def next_result(self) -> dict[str, Any]:
        return await self._outgoing.get()

    def close(self) -> None:
        self._stop.set()
        if self._worker.is_alive():
            self._worker.join(timeout=0.1)

    def _publish(self, message: dict[str, Any]) -> None:
        if self._stop.is_set():
            return
        try:
            self._loop.call_soon_threadsafe(self._enqueue_result, message)
        except RuntimeError:
            LOGGER.debug("Live connection event loop is already closed.")

    def _enqueue_result(self, message: dict[str, Any]) -> None:
        if self._stop.is_set():
            return
        if self._outgoing.full():
            try:
                self._outgoing.get_nowait()
            except asyncio.QueueEmpty:
                pass
        self._outgoing.put_nowait(message)

    def _process_frames(self) -> None:
        while not self._stop.is_set():
            payload = self._frames.get(timeout=0.1)
            if payload is None:
                continue
            try:
                frame = decode_jpeg_frame(
                    payload,
                    self._settings.max_live_frame_pixels,
                )
                inference_started = time.perf_counter()
                prediction = self._model_service.predict(
                    frame,
                    wait_for_available=True,
                )
                inference_ms = (time.perf_counter() - inference_started) * 1000
                timestamp = datetime.now(timezone.utc).isoformat()
                with self._alert_settings_lock:
                    smoothed_confidence, alert, alert_started = self._smoother.update(
                        prediction.violence_probability,
                        time.monotonic() - self._started_at,
                    )
                    alert_threshold = self._smoother.threshold
                if alert_started:
                    self._alert_store.record(
                        timestamp=timestamp,
                        source="live",
                        source_name="Live webcam",
                        label=prediction.predicted_class,
                        confidence=prediction.confidence,
                        smoothed_confidence=smoothed_confidence,
                        threshold=alert_threshold,
                        snapshot_frame=frame,
                    )

                self._publish(
                    {
                        "label": prediction.predicted_class,
                        "confidence": prediction.confidence,
                        "smoothed_confidence": smoothed_confidence,
                        "alert": alert,
                        "inference_ms": inference_ms,
                        "timestamp": timestamp,
                    }
                )
            except ValueError as exc:
                LOGGER.warning("Rejected invalid live frame: %s", exc)
                self._publish({"error": str(exc)})
            except (RuntimeError, OSError) as exc:
                LOGGER.exception("Live frame processing failed.")
                self._publish({"error": str(exc)})


class LiveSessionManager:
    def __init__(
        self,
        model_service: ModelService,
        settings: Settings,
        alert_store: AlertStore,
    ) -> None:
        self._model_service = model_service
        self._settings = settings
        self.alert_store = alert_store
        self._sessions: set[LiveSession] = set()
        self._lock = threading.Lock()

    @property
    def max_frame_size_bytes(self) -> int:
        return self._settings.max_live_frame_size_bytes

    def open_session(self, loop: asyncio.AbstractEventLoop) -> LiveSession:
        with self._lock:
            if len(self._sessions) >= self._settings.max_live_clients:
                raise RuntimeError("The live-stream connection limit has been reached.")
            session = LiveSession(
                model_service=self._model_service,
                settings=self._settings,
                alert_store=self.alert_store,
                loop=loop,
            )
            self._sessions.add(session)
            session.start()
            return session

    def close_session(self, session: LiveSession) -> None:
        session.close()
        with self._lock:
            self._sessions.discard(session)

    def close(self) -> None:
        with self._lock:
            sessions = list(self._sessions)
        for session in sessions:
            self.close_session(session)
