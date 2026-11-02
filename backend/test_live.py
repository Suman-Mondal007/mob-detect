import csv
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.app import main
from backend.app.config import Settings
from backend.app.model_service import ModelService
from backend.app.schemas import PredictionResponse
from backend.app.live_stream import AlertStore, LatestFrameSlot, decode_jpeg_frame


def _jpeg_frame(width: int = 32, height: int = 24) -> bytes:
    encoded, buffer = cv2.imencode(
        ".jpg",
        np.full((height, width, 3), 127, dtype=np.uint8),
    )
    if not encoded:
        raise RuntimeError("Could not encode test JPEG.")
    return buffer.tobytes()


class LatestFrameSlotTests(unittest.TestCase):
    def test_new_frame_replaces_queued_frame(self) -> None:
        slot = LatestFrameSlot()
        slot.put(b"old")
        slot.put(b"new")

        self.assertEqual(slot.get(timeout=0), b"new")
        self.assertEqual(slot.dropped_frames, 1)

    def test_frame_decoder_checks_jpeg_and_pixel_limit(self) -> None:
        frame = decode_jpeg_frame(_jpeg_frame(), max_pixels=32 * 24)
        self.assertEqual(frame.shape, (24, 32, 3))

        with self.assertRaisesRegex(ValueError, "valid JPEG"):
            decode_jpeg_frame(b"not an image", max_pixels=100)
        with self.assertRaisesRegex(ValueError, "pixel limit"):
            decode_jpeg_frame(_jpeg_frame(), max_pixels=10)


class AlertStoreTests(unittest.TestCase):
    def test_persists_alerts_and_snapshots_across_store_instances(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database_path = root / "nested" / "alerts.db"
            snapshots_dir = root / "snapshots"
            store = AlertStore(database_path, snapshots_dir)
            recorded = store.record(
                timestamp=datetime.now(timezone.utc).isoformat(),
                source="live",
                source_name="Front camera",
                label="Violence",
                confidence=0.9,
                smoothed_confidence=0.85,
                threshold=0.8,
                snapshot_frame=np.zeros((24, 32, 3), dtype=np.uint8),
            )
            store.close()

            reopened = AlertStore(database_path, snapshots_dir)
            response = reopened.list_alerts(page=1, page_size=10)
            self.assertEqual(response.total, 1)
            self.assertEqual(response.alerts[0].id, recorded.id)
            self.assertEqual(response.alerts[0].source, "live")
            self.assertEqual(response.alerts[0].source_name, "Front camera")
            self.assertEqual(response.stats.total_alerts, 1)
            self.assertEqual(response.stats.last_24h, 1)
            self.assertAlmostEqual(response.stats.average_confidence, 0.9)
            self.assertIsNotNone(response.alerts[0].snapshot_url)
            snapshot_path = reopened.get_snapshot_path(recorded.id)
            self.assertIsNotNone(snapshot_path)
            self.assertTrue(snapshot_path.is_file())
            rows = list(csv.DictReader(reopened.csv_content().splitlines()))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["label"], "Violence")
            reopened.close()

    def test_lists_alerts_with_source_search_and_pagination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = AlertStore(root / "alerts.db", root / "snapshots")
            for source, source_name, label in [
                ("live", "North camera", "Violence"),
                ("upload", "meeting.mp4", "Violence"),
                ("live", "Lobby camera", "NonViolence"),
            ]:
                store.record(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    source=source,
                    source_name=source_name,
                    label=label,
                    confidence=0.7,
                    smoothed_confidence=0.65,
                    threshold=0.8,
                )

            response = store.list_alerts(
                page=1,
                page_size=1,
                source="live",
                search="North",
            )
            self.assertEqual(response.total, 1)
            self.assertEqual(response.alerts[0].source_name, "North camera")
            self.assertEqual(response.total_pages, 1)
            store.close()


class LiveWebSocketTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.settings = Settings(
            _env_file=None,
            alerts_database_path=Path(self.temporary_directory.name) / "alerts.db",
            alert_snapshots_dir=Path(self.temporary_directory.name) / "snapshots",
            video_window_size=1,
            violence_threshold=0.5,
            video_alert_duration_seconds=0,
            max_live_frame_size_bytes=1024,
        )
        self.settings_patch = patch(
            "backend.app.main.get_settings",
            return_value=self.settings,
        )
        self.settings_patch.start()
        self.model_load_patch = patch.object(
            ModelService,
            "load",
            self._load_test_model,
        )
        self.model_load_patch.start()
        self.prediction_patch = patch.object(
            ModelService,
            "predict",
            self._predict_violence,
        )
        self.prediction_patch.start()
        self.client_context = TestClient(main.app)
        self.client = self.client_context.__enter__()
        token = self.client.post("/auth/login", json={"username": "admin", "password": "admin"}).json()["access_token"]
        self.client.headers.update({"Authorization": f"Bearer {token}"})

    @staticmethod
    def _load_test_model(service: ModelService) -> None:
        service._model = object()
        service._device = "cpu"
        service._device_name = "cpu"

    @staticmethod
    def _predict_violence(
        service: ModelService,
        frame: np.ndarray,
        *,
        wait_for_available: bool,
    ) -> PredictionResponse:
        if not service.is_loaded or frame.shape != (24, 32, 3):
            raise AssertionError("Unexpected live inference service or frame.")
        if not wait_for_available:
            raise AssertionError("Live predictions should wait for the model worker.")
        return PredictionResponse(
            predicted_class="Violence",
            confidence=0.9,
            violence_probability=0.9,
        )

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        self.prediction_patch.stop()
        self.model_load_patch.stop()
        self.settings_patch.stop()
        self.temporary_directory.cleanup()

    def test_websocket_returns_prediction_and_logs_alert(self) -> None:
        with self.client.websocket_connect("/ws/live") as websocket:
            websocket.send_bytes(_jpeg_frame())
            result = websocket.receive_json()

        self.assertEqual(
            set(result),
            {
                "label",
                "confidence",
                "smoothed_confidence",
                "alert",
                "inference_ms",
                "timestamp",
            },
        )
        self.assertEqual(result["label"], "Violence")
        self.assertEqual(result["confidence"], 0.9)
        self.assertEqual(result["smoothed_confidence"], 0.9)
        self.assertTrue(result["alert"])
        self.assertGreaterEqual(result["inference_ms"], 0)

        alerts = self.client.get("/alerts").json()["alerts"]
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["label"], "Violence")
        self.assertEqual(alerts[0]["source"], "live")
        self.assertEqual(alerts[0]["source_name"], "Live webcam")
        self.assertIsNotNone(alerts[0]["snapshot_url"])
        snapshot_response = self.client.get(alerts[0]["snapshot_url"])
        self.assertEqual(snapshot_response.status_code, 200)
        self.assertEqual(snapshot_response.headers["content-type"], "image/jpeg")

        filtered_response = self.client.get(
            "/alerts",
            params={
                "page": 1,
                "page_size": 1,
                "source": "live",
                "search": "webcam",
            },
        )
        self.assertEqual(filtered_response.status_code, 200)
        filtered = filtered_response.json()
        self.assertEqual(filtered["total"], 1)
        self.assertEqual(filtered["page"], 1)
        self.assertEqual(filtered["page_size"], 1)
        self.assertEqual(len(filtered["alerts"]), 1)

        csv_response = self.client.get("/alerts.csv")
        self.assertEqual(csv_response.status_code, 200)
        self.assertEqual(csv_response.headers["content-type"], "text/csv; charset=utf-8")
        self.assertIn("Violence", csv_response.text)
        filtered_csv = self.client.get(
            "/alerts.csv",
            params={"source": "upload"},
        )
        self.assertEqual(filtered_csv.status_code, 200)
        self.assertNotIn("Violence", filtered_csv.text)
        missing_snapshot = self.client.get("/alerts/999/snapshot")
        self.assertEqual(missing_snapshot.status_code, 404)

    def test_websocket_reports_invalid_jpeg(self) -> None:
        with self.client.websocket_connect("/ws/live") as websocket:
            websocket.send_bytes(b"not a jpeg")
            result = websocket.receive_json()

        self.assertEqual(result, {"error": "Live frame is not a valid JPEG image."})

    def test_websocket_closes_when_frame_exceeds_size_limit(self) -> None:
        with self.client.websocket_connect("/ws/live") as websocket:
            websocket.send_bytes(b"x" * (self.settings.max_live_frame_size_bytes + 1))
            with self.assertRaises(WebSocketDisconnect) as error:
                websocket.receive_json()

        self.assertEqual(error.exception.code, 1009)

    def test_alert_endpoint_starts_with_empty_results(self) -> None:
        response_body = self.client.get("/alerts").json()
        self.assertEqual(response_body["alerts"], [])
        self.assertEqual(response_body["total"], 0)
        self.assertEqual(response_body["stats"]["total_alerts"], 0)
        response = self.client.get("/alerts.csv")
        self.assertEqual(response.status_code, 200)
        self.assertIn("smoothed_confidence", response.text)


if __name__ == "__main__":
    unittest.main()
