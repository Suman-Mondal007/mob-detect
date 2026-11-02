import base64
import threading

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.camera_store import Camera, CameraStore
from backend.app.config import Settings
from backend.app.model_service import ModelService
from backend.app.schemas import PredictionResponse
from backend.seed_demo_cameras import seed_demo_cameras


class _FakeCapture:
    def __init__(self, _source, *_args) -> None:
        self._released = threading.Event()
        self._frame = np.full((24, 32, 3), 127, dtype=np.uint8)

    def isOpened(self) -> bool:
        return not self._released.is_set()

    def set(self, _property: int, _value: float) -> bool:
        return True

    def get(self, _property: int) -> float:
        return 10

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._released.is_set():
            return False, None
        return True, self._frame.copy()

    def release(self) -> None:
        self._released.set()


def test_camera_store_adds_persistent_camera_table(tmp_path) -> None:
    database_path = tmp_path / "cameras.db"
    store = CameraStore(database_path)
    original = store.create(
        Camera(
            id="front-door",
            name="Front Door",
            location="North entrance",
            source_type="webcam",
            source_uri="0",
            enabled=False,
        )
    )
    assert original.id == "front-door"
    assert store.get("front-door").location == "North entrance"

    updated = store.update(
        "front-door",
        name="Updated Front Door",
        location="East entrance",
        source_type="webcam",
        source_uri="1",
        enabled=True,
    )
    assert updated is not None
    assert updated.name == "Updated Front Door"
    store.close()

    reopened = CameraStore(database_path)
    restored = reopened.get("front-door")
    assert restored is not None
    assert restored.source_uri == "1"
    assert restored.enabled
    assert reopened.delete("front-door")
    assert not reopened.delete("front-door")
    reopened.close()


def test_seed_script_creates_six_looped_demo_camera_records(tmp_path) -> None:
    videos = tmp_path / "demo_videos"
    videos.mkdir()
    clip = videos / "sample.mp4"
    clip.touch()
    store = CameraStore(tmp_path / "cameras.db")
    try:
        assert seed_demo_cameras(videos, store) == 6
        cameras = store.list()
        assert len(cameras) == 6
        assert [camera.id for camera in cameras] == [
            f"demo-{index:02d}" for index in range(1, 7)
        ]
        assert all(camera.source_uri == str(clip.resolve()) for camera in cameras)
        assert all(camera.source_type == "file" and camera.enabled for camera in cameras)
    finally:
        store.close()


def test_camera_crud_test_snapshot_and_credential_masking(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr("backend.app.camera_manager.cv2.VideoCapture", _FakeCapture)
    settings = Settings(
        _env_file=None,
        alerts_database_path=tmp_path / "alerts.db",
        alert_snapshots_dir=tmp_path / "snapshots",
        camera_inference_fps=10,
        camera_total_inference_fps=10,
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
                confidence=0.9,
                violence_probability=0.1,
            )
            for _image in images
        ]

    monkeypatch.setattr(ModelService, "load", load_model)
    monkeypatch.setattr(ModelService, "predict_batch", predict_batch)
    with TestClient(main.app) as client:
        token = client.post("/auth/login", json={"username": "admin", "password": "admin"}).json()["access_token"]
        client.headers.update({"Authorization": f"Bearer {token}"})
        token = client.post("/auth/login", json={"username": "admin", "password": "admin"}).json()["access_token"]
        client.headers.update({"Authorization": f"Bearer {token}"})
        created = client.post(
            "/cameras",
            json={
                "id": "lobby",
                "name": "Lobby",
                "location": "Main building",
                "source_type": "rtsp",
                "source_uri": "rtsp://operator:secret@example.test/live",
            },
        )
        assert created.status_code == 201
        assert created.json()["id"] == "lobby"
        assert created.json()["source_type"] == "rtsp"
        assert created.json()["source_uri"] == "rtsp://***@example.test"
        assert "secret" not in created.text

        cameras = client.get("/cameras")
        assert cameras.status_code == 200
        assert cameras.json()["cameras"][0]["name"] == "Lobby"
        assert "secret" not in cameras.text
        assert client.get("/cameras/lobby").json()["name"] == "Lobby"
        status = client.get("/cameras/status")
        assert status.status_code == 200
        assert status.json()["cameras"][0]["camera_id"] == "lobby"

        tested = client.post("/cameras/lobby/test")
        assert tested.status_code == 200
        preview = tested.json()
        assert preview["ok"]
        assert preview["preview_mime_type"] == "image/jpeg"
        assert base64.b64decode(preview["preview_frame_base64"]).startswith(b"\xff\xd8")

        snapshot = client.get("/cameras/lobby/snapshot")
        assert snapshot.status_code == 200
        assert snapshot.headers["content-type"] == "image/jpeg"

        updated = client.put(
            "/cameras/lobby",
            json={"name": "Updated lobby", "enabled": False},
        )
        assert updated.status_code == 200
        assert updated.json()["name"] == "Updated lobby"
        assert updated.json()["enabled"] is False
        assert client.get("/cameras/lobby/snapshot").status_code == 409

        assert client.delete("/cameras/lobby").status_code == 204
        assert client.get("/cameras").json()["cameras"] == []
        assert client.post("/cameras/lobby/test").status_code == 404


def test_camera_test_returns_source_errors_without_server_failure(
    monkeypatch,
    tmp_path,
) -> None:
    class ClosedCapture(_FakeCapture):
        def isOpened(self) -> bool:
            return False

    monkeypatch.setattr("backend.app.camera_manager.cv2.VideoCapture", ClosedCapture)
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
        token = client.post("/auth/login", json={"username": "admin", "password": "admin"}).json()["access_token"]
        client.headers.update({"Authorization": f"Bearer {token}"})
        created = client.post(
            "/cameras",
            json={
                "id": "offline",
                "name": "Offline camera",
                "source_type": "webcam",
                "source_uri": "0",
                "enabled": False,
            },
        )
        assert created.status_code == 201

        tested = client.post("/cameras/offline/test")
        assert tested.status_code == 200
        assert tested.json() == {
            "ok": False,
            "error": "Could not open camera source.",
            "preview_frame_base64": None,
            "preview_mime_type": None,
        }
