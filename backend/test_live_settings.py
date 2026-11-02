import json
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.config import Settings
from backend.app.model_service import ModelService
from backend.app.schemas import PredictionResponse


def _jpeg_frame() -> bytes:
    encoded, buffer = cv2.imencode(
        ".jpg",
        np.full((24, 32, 3), 127, dtype=np.uint8),
    )
    if not encoded:
        raise RuntimeError("Could not encode test JPEG.")
    return buffer.tobytes()


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    settings = Settings(
        _env_file=None,
        alerts_database_path=tmp_path / "alerts.db",
        alert_snapshots_dir=tmp_path / "snapshots",
        violence_threshold=0.99,
        video_alert_duration_seconds=2,
        video_window_size=1,
        max_live_frame_size_bytes=1024,
    )
    monkeypatch.setattr(main, "get_settings", lambda: settings)

    def load_model(service: ModelService) -> None:
        service._model = object()
        service._device = "cpu"
        service._device_name = "cpu"

    def predict(
        service: ModelService,
        frame: np.ndarray,
        *,
        wait_for_available: bool,
    ) -> PredictionResponse:
        assert service.is_loaded
        assert frame.shape == (24, 32, 3)
        assert wait_for_available
        return PredictionResponse(
            predicted_class="Violence",
            confidence=0.9,
            violence_probability=0.9,
        )

    monkeypatch.setattr(ModelService, "load", load_model)
    monkeypatch.setattr(ModelService, "predict", predict)
    with TestClient(main.app) as client:
        token = client.post("/auth/login", json={"username": "admin", "password": "admin"}).json()["access_token"]
        client.headers.update({"Authorization": f"Bearer {token}"})
        yield client


def test_live_settings_control_threshold_and_sensitivity(api_client) -> None:
    with api_client.websocket_connect("/ws/live") as websocket:
        websocket.send_text(
            json.dumps({"threshold": 0.7, "sensitivity": "high"})
        )
        websocket.send_bytes(_jpeg_frame())
        initial = websocket.receive_json()
        assert initial["alert"] is False

        time.sleep(0.55)
        websocket.send_bytes(_jpeg_frame())
        triggered = websocket.receive_json()

    assert triggered["alert"] is True
    alerts = api_client.get("/alerts").json()["alerts"]
    assert len(alerts) == 1
    assert alerts[0]["threshold"] == pytest.approx(0.7)


def test_invalid_live_settings_return_a_clear_error(api_client) -> None:
    with api_client.websocket_connect("/ws/live") as websocket:
        websocket.send_text(json.dumps({"threshold": 1.5, "sensitivity": "high"}))
        response = websocket.receive_json()

    assert response == {
        "error": (
            "Live settings must include a threshold from 0 to 1 "
            "and sensitivity low, balanced, or high."
        )
    }
