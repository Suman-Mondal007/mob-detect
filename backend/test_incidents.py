import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.alerts_store import AlertStore
from backend.app.camera_manager import CameraManager
from backend.app.config import Settings
from backend.app.incidents_store import IncidentStore
from backend.app.model_service import ModelService
from backend.app.schemas import PredictionResponse


class _FakeCapture:
    def __init__(self, _source, *_args) -> None:
        self._closed = threading.Event()
        self._number = 0

    def isOpened(self) -> bool:
        return not self._closed.is_set()

    def set(self, _property: int, _value: float) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._closed.wait(0.005):
            return False, None
        self._number += 1
        return True, np.full((120, 160, 3), self._number % 255, dtype=np.uint8)

    def release(self) -> None:
        self._closed.set()


class _FakeModel:
    is_loaded = True

    def __init__(self, probability: list[float]) -> None:
        self.probability = probability

    def predict_batch(
        self,
        images: list[np.ndarray],
        *,
        wait_for_available: bool,
    ) -> list[PredictionResponse]:
        assert wait_for_available
        value = self.probability[0]
        return [
            PredictionResponse(
                predicted_class="Violence" if value >= 0.5 else "NonViolence",
                confidence=max(value, 1 - value),
                violence_probability=value,
            )
            for _image in images
        ]


def _settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "_env_file": None,
        "alerts_database_path": tmp_path / "alerts.db",
        "alert_snapshots_dir": tmp_path / "alert-snapshots",
        "incident_evidence_dir": tmp_path / "incident-evidence",
        "incident_retention_days": 30,
        "incident_ring_buffer_seconds": 1,
        "incident_evidence_fps": 10,
        "camera_inference_fps": 20,
        "camera_total_inference_fps": 20,
        "video_window_size": 1,
        "violence_threshold": 0.5,
        "video_alert_duration_seconds": 0,
        "camera_alert_cooldown_seconds": 0.02,
    }
    values.update(overrides)
    return Settings(**values)


def _wait_for(predicate, timeout: float = 4) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return bool(predicate())


def test_incident_store_persists_metadata_clip_and_expires_evidence(
    tmp_path: Path,
) -> None:
    store = IncidentStore(
        tmp_path / "incidents.db",
        tmp_path / "evidence",
        retention_days=1,
        ring_buffer_seconds=1,
        evidence_fps=5,
        cleanup_interval_seconds=3600,
    )
    try:
        started = datetime.now(timezone.utc)
        incident = store.create_incident(
            camera_id="lobby",
            start_time=started.isoformat(),
            peak_confidence=0.8,
            snapshot_frame=np.full((48, 64, 3), 127, dtype=np.uint8),
        )
        snapshot_path = store.get_snapshot_path(incident.id)
        assert snapshot_path is not None and snapshot_path.is_file()
        store.update_peak(incident.id, 0.95)
        store.update_peak(incident.id, 0.7)
        store.close_incident(
            incident.id,
            end_time=(started + timedelta(seconds=2)).isoformat(),
            frames=[
                (1.0, np.full((48, 64, 3), 64, dtype=np.uint8)),
                (1.2, np.full((48, 64, 3), 128, dtype=np.uint8)),
                (1.4, np.full((48, 64, 3), 200, dtype=np.uint8)),
            ],
        )
        assert _wait_for(
            lambda: (store.get_incident(incident.id) or incident).clip_url is not None
        )
        saved = store.get_incident(incident.id)
        assert saved is not None
        assert saved.peak_confidence == 0.95
        assert saved.end_time is not None
        clip_path = store.get_clip_path(incident.id)
        assert clip_path is not None and clip_path.is_file() and clip_path.stat().st_size
        second = store.create_incident(
            camera_id="loading-bay",
            start_time=(started + timedelta(seconds=3)).isoformat(),
            peak_confidence=0.6,
            snapshot_frame=np.full((48, 64, 3), 127, dtype=np.uint8),
        )
        page_one = store.list_incidents(page=1, page_size=1)
        page_two = store.list_incidents(page=2, page_size=1)
        assert page_one.total == 2 and page_one.total_pages == 2
        assert page_one.incidents[0].id == second.id
        assert page_two.incidents[0].id == incident.id
        assert store.list_incidents(camera_id="lobby", status="new").total == 1
        assert "camera_id" in store.csv_content()

        assert store.cleanup_expired_evidence(
            now=started + timedelta(days=1, seconds=2)
        ) == 2
        expired = store.get_incident(incident.id)
        assert expired is not None
        assert expired.snapshot_url is None
        assert expired.clip_url is None
        assert store.get_snapshot_path(incident.id) is None
        assert store.get_clip_path(incident.id) is None
        assert store.get_snapshot_path(second.id) is not None
    finally:
        store.close()


def test_camera_alert_transition_creates_updates_and_closes_incident(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("backend.app.camera_manager.cv2.VideoCapture", _FakeCapture)
    settings = _settings(tmp_path)
    probability = [0.8]
    incident_store = IncidentStore(
        settings.resolved_alerts_database_path,
        settings.resolved_incident_evidence_dir,
        retention_days=settings.incident_retention_days,
        ring_buffer_seconds=settings.incident_ring_buffer_seconds,
        evidence_fps=settings.incident_evidence_fps,
        cleanup_interval_seconds=3600,
    )
    alert_store = AlertStore(
        settings.resolved_alerts_database_path,
        settings.resolved_alert_snapshots_dir,
    )
    manager = CameraManager(
        settings,
        _FakeModel(probability),
        alert_store,
        incident_store,
    )
    try:
        manager.add_camera("entry", 0)
        assert _wait_for(lambda: incident_store.list_incidents().total == 1)
        first = incident_store.list_incidents().incidents[0]
        assert first.status == "new"
        assert first.snapshot_url is not None

        probability[0] = 0.97
        assert _wait_for(
            lambda: (incident_store.get_incident(first.id) or first).peak_confidence
            == 0.97
        )
        probability[0] = 0.1
        assert _wait_for(lambda: manager.list_cameras()[0].state == "COOLDOWN")
        assert _wait_for(
            lambda: (incident_store.get_incident(first.id) or first).end_time is not None
        )
        assert _wait_for(
            lambda: (incident_store.get_incident(first.id) or first).clip_url is not None
        )
        assert incident_store.get_clip_path(first.id) is not None
    finally:
        manager.close()
        incident_store.close()
        alert_store.close()


def test_incident_api_filters_actions_and_serves_evidence(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("backend.app.camera_manager.cv2.VideoCapture", _FakeCapture)
    settings = _settings(tmp_path)
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    probability = [0.9]

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
        value = probability[0]
        return [
            PredictionResponse(
                predicted_class="Violence" if value >= 0.5 else "NonViolence",
                confidence=max(value, 1 - value),
                violence_probability=value,
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
                "id": "incident-entry",
                "name": "Incident entry",
                "source_type": "webcam",
                "source_uri": "0",
            },
        )
        assert created.status_code == 201
        assert _wait_for(lambda: client.get("/incidents").json()["total"] == 1)
        listed = client.get(
            "/incidents",
            params={"camera_id": "incident-entry", "status": "new", "page_size": 1},
        )
        assert listed.status_code == 200
        incident = listed.json()["incidents"][0]
        incident_id = incident["id"]
        assert incident["peak_confidence"] == 0.9
        assert client.get(f"/incidents/{incident_id}").status_code == 200
        assert client.get(incident["snapshot_url"]).headers["content-type"] == "image/jpeg"
        assert client.get(f"/incidents?status=invalid").status_code == 422
        assert client.get("/incidents?start_time=2025-01-02T00:00:00Z&end_time=2025-01-01T00:00:00Z").status_code == 422
        assert client.get(
            "/incidents?start_time=2025-01-02T00:00:00"
            "&end_time=2025-01-01T00:00:00Z"
        ).status_code == 422
        export = client.get("/incidents/export.csv", params={"camera_id": "incident-entry"})
        assert export.status_code == 200
        assert "camera_id" in export.text

        acknowledged = client.post(f"/incidents/{incident_id}/acknowledge")
        assert acknowledged.status_code == 200
        assert acknowledged.json()["status"] == "acknowledged"
        marked = client.post(f"/incidents/{incident_id}/false-positive")
        assert marked.status_code == 200
        assert marked.json()["status"] == "false-positive"
        assert client.post(f"/incidents/{incident_id}/acknowledge").status_code == 409

        probability[0] = 0.1
        assert _wait_for(
            lambda: client.get("/cameras/status").json()["cameras"][0]["state"]
            == "COOLDOWN"
        )
        assert _wait_for(
            lambda: client.get(f"/incidents/{incident_id}").json()["clip_url"] is not None
        )
        clip = client.get(f"/incidents/{incident_id}/clip")
        assert clip.status_code == 200
        assert clip.headers["content-type"] == "video/mp4"
        assert client.get("/incidents/999999").status_code == 404

    # App shutdown closes the camera incident and drains the asynchronous MP4 writer.
    clip_path = settings.resolved_incident_evidence_dir
    assert any(clip_path.glob("incident-*.mp4"))
