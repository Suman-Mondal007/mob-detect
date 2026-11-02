import asyncio
import threading
import time
from datetime import datetime, timezone

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.camera_manager import CameraFrame, CameraWorker
from backend.app.config import Settings
from backend.app.model_service import ModelService
from backend.app.schemas import PredictionResponse


class _FakeCapture:
    def __init__(self, _source, *_args) -> None:
        self._closed = threading.Event()
        self._frame = np.full((300, 1200, 3), 127, dtype=np.uint8)

    def isOpened(self) -> bool:
        return not self._closed.is_set()

    def set(self, _property: int, _value: float) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._closed.is_set():
            return False, None
        return True, self._frame.copy()

    def release(self) -> None:
        self._closed.set()


class _FrameManager:
    def __init__(self) -> None:
        self.frame = CameraFrame(
            image=np.full((160, 640, 3), 127, dtype=np.uint8),
            captured_at=datetime.now(timezone.utc),
            captured_monotonic=time.monotonic(),
            display_image=np.full((300, 1200, 3), 127, dtype=np.uint8),
        )

    def latest_frame(self, _camera_id: str) -> CameraFrame:
        return self.frame


def _decode_mjpeg_part(part: bytes) -> np.ndarray:
    payload = part.split(b"\r\n\r\n", maxsplit=1)[1].removesuffix(b"\r\n")
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise AssertionError("MJPEG part did not contain a valid JPEG image.")
    return image


def test_grid_and_full_streams_downscale_to_expected_widths() -> None:
    async def read_first_frame(quality: str) -> np.ndarray:
        generator = main._mjpeg_stream(
            _FrameManager(),
            "camera",
            quality=quality,
        )
        try:
            part = await generator.__anext__()
        finally:
            await generator.aclose()
        return _decode_mjpeg_part(part)

    grid = asyncio.run(read_first_frame("grid"))
    full = asyncio.run(read_first_frame("full"))
    assert grid.shape[1] == 480
    assert full.shape[1] == 960


def test_stream_output_is_capped_at_15_fps() -> None:
    async def read_two_frames() -> float:
        generator = main._mjpeg_stream(
            _FrameManager(),
            "camera",
            quality="grid",
        )
        try:
            await generator.__anext__()
            first_frame_at = time.monotonic()
            await generator.__anext__()
            return time.monotonic() - first_frame_at
        finally:
            await generator.aclose()

    assert asyncio.run(read_two_frames()) >= 0.05


def test_worker_keeps_display_and_inference_frames_at_separate_sizes(
    tmp_path,
) -> None:
    settings = Settings(
        _env_file=None,
        alerts_database_path=tmp_path / "alerts.db",
        alert_snapshots_dir=tmp_path / "snapshots",
        camera_frame_width=640,
        camera_frame_height=360,
        camera_display_frame_width=960,
        camera_display_frame_height=960,
    )
    worker = CameraWorker(
        camera_id="resolution-check",
        source=0,
        source_type="webcam",
        settings=settings,
    )
    worker._publish_frame(np.full((600, 1200, 3), 127, dtype=np.uint8))

    frame = worker.latest_frame()
    assert frame is not None
    assert frame.image.shape[:2] == (320, 640)
    assert frame.display_image is not None
    assert frame.display_image.shape[:2] == (480, 960)


def test_events_websocket_broadcasts_status_and_alert_then_cleans_up(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr("backend.app.camera_manager.cv2.VideoCapture", _FakeCapture)
    monkeypatch.setattr(main, "EVENT_HEARTBEAT_SECONDS", 0.2)
    settings = Settings(
        _env_file=None,
        alerts_database_path=tmp_path / "alerts.db",
        alert_snapshots_dir=tmp_path / "snapshots",
        camera_inference_fps=20,
        camera_total_inference_fps=20,
        video_window_size=1,
        violence_threshold=0.5,
        video_alert_duration_seconds=0,
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)

    def load_model(service: ModelService) -> None:
        service._model = object()
        service._device = "cpu"
        service._device_name = "cpu"

    def predict_batch(
        _service: ModelService,
        images: list[np.ndarray],
        *,
        wait_for_available: bool,
    ) -> list[PredictionResponse]:
        assert wait_for_available
        return [
            PredictionResponse(
                predicted_class="Violence",
                confidence=0.9,
                violence_probability=0.9,
            )
            for _image in images
        ]

    monkeypatch.setattr(ModelService, "load", load_model)
    monkeypatch.setattr(ModelService, "predict_batch", predict_batch)
    with TestClient(main.app) as client:
        token = client.post("/auth/login", json={"username": "admin", "password": "admin"}).json()["access_token"]
        client.headers.update({"Authorization": f"Bearer {token}"})
        with client.websocket_connect("/ws/events") as websocket:
            assert websocket.receive_json()["type"] == "heartbeat"
            created = client.post(
                "/cameras",
                json={
                    "id": "events-entry",
                    "name": "Events entry",
                    "source_type": "webcam",
                    "source_uri": "0",
                },
            )
            assert created.status_code == 201

            received_alert = False
            for _ in range(10):
                event = websocket.receive_json()
                if event["type"] == "camera_status":
                    camera = next(
                        (
                            item
                            for item in event["cameras"]
                            if item["camera_id"] == "events-entry"
                        ),
                        None,
                    )
                    if camera is not None:
                        assert {
                            "state",
                            "confidence",
                            "online",
                            "inference_ms",
                        }.issubset(camera)
                elif event["type"] == "alert":
                    assert event["camera_id"] == "events-entry"
                    received_alert = True
                    break
            assert received_alert

        assert not client.app.state.camera_event_hub._subscribers


def test_stream_route_rejects_unknown_camera_and_invalid_quality(
    monkeypatch,
    tmp_path,
) -> None:
    settings = Settings(
        _env_file=None,
        alerts_database_path=tmp_path / "alerts.db",
        alert_snapshots_dir=tmp_path / "snapshots",
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)

    def load_model(service: ModelService) -> None:
        service._model = object()
        service._device = "cpu"
        service._device_name = "cpu"

    monkeypatch.setattr(ModelService, "load", load_model)
    with TestClient(main.app) as client:
        token = client.post("/auth/login", json={"username": "admin", "password": "admin"}).json()["access_token"]
        client.headers.update({"Authorization": f"Bearer {token}"})
        page = client.get("/stream-test")
        assert page.status_code == 200
        assert "six-camera stream test" in page.text
        assert client.get("/stream/missing").status_code == 404
        assert client.get(
            "/stream/missing?quality=bad",
        ).status_code == 422
