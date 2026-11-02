import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.camera_manager import (
    CameraManager,
    _normalize_source,
    _safe_source_display,
)
from backend.app.config import Settings
from backend.app.model_service import ModelService
from backend.app.schemas import PredictionResponse


class _FakeCapture:
    def __init__(self, _source: str | int, *_args: object) -> None:
        self._released = threading.Event()
        self._frame_number = 0

    def isOpened(self) -> bool:
        return not self._released.is_set()

    def set(self, _property: int, _value: float) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._released.wait(0.01):
            return False, None
        self._frame_number += 1
        return True, np.full((24, 32, 3), self._frame_number % 255, dtype=np.uint8)

    def release(self) -> None:
        self._released.set()


class _FakeModelService:
    is_loaded = True

    def __init__(self) -> None:
        self.inference_shapes: list[tuple[int, ...]] = []
        self.batch_sizes: list[int] = []
        self._lock = threading.Lock()

    def predict(
        self,
        image: np.ndarray,
        *,
        wait_for_available: bool,
    ) -> PredictionResponse:
        return self.predict_batch(
            [image],
            wait_for_available=wait_for_available,
        )[0]

    def predict_batch(
        self,
        images: list[np.ndarray],
        *,
        wait_for_available: bool,
    ) -> list[PredictionResponse]:
        assert wait_for_available
        with self._lock:
            self.inference_shapes.extend(image.shape for image in images)
            self.batch_sizes.append(len(images))
        return [
            PredictionResponse(
                predicted_class="NonViolence",
                confidence=0.95,
                violence_probability=0.05,
            )
            for _ in images
        ]


def test_normalize_camera_source_supports_webcam_url_and_file(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.touch()

    assert _normalize_source(2) == (2, "webcam")
    assert _normalize_source("rtsp://operator:secret@example.com/live") == (
        "rtsp://operator:secret@example.com/live",
        "url",
    )
    normalized_path, source_type = _normalize_source(str(video_path))
    assert normalized_path == str(video_path.resolve())
    assert source_type == "file"

    with pytest.raises(ValueError, match="does not exist"):
        _normalize_source(str(tmp_path / "missing.mp4"))
    with pytest.raises(ValueError, match="negative"):
        _normalize_source(-1)


def test_camera_status_masks_stream_credentials() -> None:
    display = _safe_source_display(
        "rtsp://operator:secret@example.com:8554/live",
        "url",
    )
    assert display == "rtsp://***@example.com:8554"
    assert "secret" not in display


def test_camera_worker_feeds_shared_inference_scheduler(monkeypatch) -> None:
    monkeypatch.setattr("backend.app.camera_manager.cv2.VideoCapture", _FakeCapture)
    settings = Settings(
        _env_file=None,
        camera_frame_width=16,
        camera_frame_height=12,
        camera_inference_fps=10,
        camera_total_inference_fps=10,
        video_window_size=1,
        violence_threshold=0.8,
    )
    model_service = _FakeModelService()
    manager = CameraManager(settings, model_service)
    try:
        manager.add_camera("lobby", 0)
        deadline = time.monotonic() + 2
        camera_status = manager.list_cameras()[0]
        while camera_status.last_inference_at is None and time.monotonic() < deadline:
            time.sleep(0.01)
            camera_status = manager.list_cameras()[0]

        assert camera_status.online
        assert camera_status.last_frame_at is not None
        assert camera_status.last_inference_at is not None
        assert camera_status.predicted_class == "NonViolence"
        assert camera_status.confidence == pytest.approx(0.95)
        assert camera_status.smoothed_confidence == pytest.approx(0.05)
        assert camera_status.inference_ms is not None
        assert model_service.inference_shapes
        assert model_service.inference_shapes[0] == (12, 16, 3)
        assert manager.remove_camera("lobby")
        assert not manager.remove_camera("lobby")
    finally:
        manager.close()


def test_camera_alert_acknowledge_and_cooldown_state_machine(monkeypatch) -> None:
    monkeypatch.setattr("backend.app.camera_manager.cv2.VideoCapture", _FakeCapture)
    settings = Settings(
        _env_file=None,
        camera_inference_fps=20,
        camera_total_inference_fps=20,
        video_window_size=1,
        violence_threshold=0.5,
        video_alert_duration_seconds=0,
        camera_alert_cooldown_seconds=0.05,
    )
    model_service = _FakeModelService()
    model_service.probability = 0.9
    model_service.predict_batch = lambda images, *, wait_for_available: [
        PredictionResponse(
            predicted_class="Violence",
            confidence=model_service.probability,
            violence_probability=model_service.probability,
        )
        for _image in images
    ]
    manager = CameraManager(settings, model_service)
    try:
        manager.add_camera("entry", 0)
        deadline = time.monotonic() + 2
        status = manager.list_cameras()[0]
        while status.state != "ALERT" and time.monotonic() < deadline:
            time.sleep(0.01)
            status = manager.list_cameras()[0]
        assert status.state == "ALERT"
        assert status.alert

        acknowledged = manager.acknowledge("entry")
        assert acknowledged is not None
        assert acknowledged.state == "ACKNOWLEDGED"
        assert not acknowledged.alert

        model_service.probability = 0.1
        deadline = time.monotonic() + 2
        status = manager.list_cameras()[0]
        while status.state != "NORMAL" and time.monotonic() < deadline:
            time.sleep(0.01)
            status = manager.list_cameras()[0]
        assert status.state == "NORMAL"
        assert manager.acknowledge("missing") is None
        with pytest.raises(ValueError, match="Only active alerts"):
            manager.acknowledge("entry")
    finally:
        manager.close()


def test_scheduler_batches_latest_frames_from_multiple_cameras(monkeypatch) -> None:
    monkeypatch.setattr("backend.app.camera_manager.cv2.VideoCapture", _FakeCapture)
    settings = Settings(
        _env_file=None,
        camera_inference_fps=20,
        camera_total_inference_fps=12,
    )
    model_service = _FakeModelService()
    predict_batch = model_service.predict_batch

    def slower_predict_batch(
        images: list[np.ndarray],
        *,
        wait_for_available: bool,
    ) -> list[PredictionResponse]:
        time.sleep(0.1)
        return predict_batch(images, wait_for_available=wait_for_available)

    model_service.predict_batch = slower_predict_batch
    manager = CameraManager(settings, model_service)
    try:
        for index in range(6):
            manager.add_camera(f"camera-{index}", index)
        deadline = time.monotonic() + 2
        while (
            max(model_service.batch_sizes, default=0) < 2
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert max(model_service.batch_sizes, default=0) > 1
        assert max(model_service.batch_sizes) <= 8
    finally:
        manager.close()


def test_camera_api_masks_credentials_and_stops_workers(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("backend.app.camera_manager.cv2.VideoCapture", _FakeCapture)
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

    def predict_batch(
        _service: ModelService,
        images: list[np.ndarray],
        *,
        wait_for_available: bool,
    ) -> list[PredictionResponse]:
        assert wait_for_available
        return [
            PredictionResponse(
                predicted_class="NonViolence",
                confidence=0.95,
                violence_probability=0.05,
            )
            for _ in images
        ]

    monkeypatch.setattr(ModelService, "load", load_model)
    monkeypatch.setattr(ModelService, "predict_batch", predict_batch)
    with TestClient(main.app) as client:
        token = client.post("/auth/login", json={"username": "admin", "password": "admin"}).json()["access_token"]
        client.headers.update({"Authorization": f"Bearer {token}"})
        response = client.post(
            "/cameras",
            json={
                "camera_id": "loading-bay",
                "source": "rtsp://operator:secret@example.com/live",
            },
        )
        assert response.status_code == 201
        assert response.json()["source"] == "rtsp://***@example.com"
        assert "secret" not in response.text

        assert client.get("/cameras").json()["cameras"][0]["camera_id"] == "loading-bay"
        assert client.delete("/cameras/loading-bay").status_code == 204
        assert client.delete("/cameras/loading-bay").status_code == 404

        invalid = client.post(
            "/cameras",
            json={"camera_id": "bad id", "source": 0},
        )
        assert invalid.status_code == 422
        invalid_source = client.post(
            "/cameras",
            json={"camera_id": "bad-source", "source": True},
        )
        assert invalid_source.status_code == 422


def test_model_service_predict_batch_returns_one_result_per_frame(monkeypatch) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from backend.app import model_service as model_service_module

    service = ModelService(Settings(_env_file=None))
    service._model = object()
    service._device = "cpu"
    service._executor = ThreadPoolExecutor(max_workers=1)
    monkeypatch.setattr(
        service,
        "_predict_on_device",
        lambda source, _device: ["first", "second"],
    )
    monkeypatch.setattr(
        model_service_module,
        "_violence_probability",
        lambda _result: ("NonViolence", 0.9, 0.1),
    )
    try:
        predictions = service.predict_batch(
            [
                np.zeros((12, 16, 3), dtype=np.uint8),
                np.ones((12, 16, 3), dtype=np.uint8),
            ],
            wait_for_available=True,
        )
        assert len(predictions) == 2
        assert all(item.predicted_class == "NonViolence" for item in predictions)
    finally:
        service.close()
