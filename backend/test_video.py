import asyncio
import base64
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import httpx
import numpy as np

from backend.app import main
from backend.app.alerts_store import AlertStore
from backend.app.config import Settings
from backend.app.model_service import ModelService
from backend.app.schemas import PredictionResponse, VideoResult
from backend.app.video_jobs import VideoJobManager, process_video


class _FakeCapture:
    def __init__(self, path: str) -> None:
        self.path = path
        self.frames = [np.zeros((360, 640, 3), dtype=np.uint8) for _ in range(6)]
        self.index = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        return {
            cv2.CAP_PROP_FPS: 2,
            cv2.CAP_PROP_FRAME_COUNT: len(self.frames),
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 360,
        }[property_id]

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self) -> None:
        self.released = True


class _FakeModelService:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = iter(probabilities)

    def predict(
        self,
        frame: np.ndarray,
        *,
        wait_for_available: bool,
    ) -> PredictionResponse:
        assert frame.shape == (360, 640, 3)
        assert wait_for_available is True
        probability = next(self.probabilities)
        return PredictionResponse(
            predicted_class="Violence",
            confidence=probability,
            violence_probability=probability,
        )


def _completed_result() -> VideoResult:
    return VideoResult(
        verdict="NO_VIOLENCE_ALERT",
        max_violence_probability=0.2,
        sampled_frames=1,
        timeline=[{"time_seconds": 0, "violence_probability": 0.2}],
        alert_ranges=[],
        key_frames=[],
    )


class ProcessVideoTests(unittest.TestCase):
    def test_samples_smooths_builds_alert_ranges_and_keeps_five_key_frames(self) -> None:
        capture = _FakeCapture("video.mp4")
        settings = Settings(
            _env_file=None,
            video_sampling_fps=1,
            video_window_size=1,
            violence_threshold=0.5,
            video_alert_duration_seconds=1,
        )
        progress: list[float] = []
        with patch("backend.app.video_jobs.cv2.VideoCapture", return_value=capture):
            result = process_video(
                Path("video.mp4"),
                _FakeModelService([0.8, 0.9, 0.2]),
                settings,
                progress.append,
            )

        self.assertTrue(capture.released)
        self.assertEqual(result.sampled_frames, 3)
        self.assertEqual(result.verdict, "VIOLENCE_ALERT")
        self.assertEqual(
            [(item.start_seconds, item.end_seconds) for item in result.alert_ranges],
            [(1, 2)],
        )
        self.assertEqual(
            [item.violence_probability for item in result.timeline],
            [0.8, 0.9, 0.2],
        )
        self.assertEqual(progress[-1], 100)
        self.assertEqual(len(result.key_frames), 3)
        encoded = result.key_frames[0].image_base64
        decoded = cv2.imdecode(
            np.frombuffer(base64.b64decode(encoded), dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.shape[1], 480)

    def test_records_upload_alerts_with_snapshot_and_video_time(self) -> None:
        capture = _FakeCapture("video.mp4")
        settings = Settings(
            _env_file=None,
            video_sampling_fps=1,
            video_window_size=1,
            violence_threshold=0.5,
            video_alert_duration_seconds=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = AlertStore(root / "alerts.db", root / "snapshots")
            try:
                with patch("backend.app.video_jobs.cv2.VideoCapture", return_value=capture):
                    process_video(
                        Path("video.mp4"),
                        _FakeModelService([0.8, 0.9, 0.2]),
                        settings,
                        lambda progress: None,
                        alert_store=store,
                        source_name="incident clip.mp4",
                    )

                alerts = store.list_alerts().alerts
                self.assertEqual(len(alerts), 1)
                self.assertEqual(alerts[0].source, "upload")
                self.assertEqual(alerts[0].source_name, "incident clip.mp4")
                self.assertEqual(alerts[0].video_time_seconds, 1)
                self.assertIsNotNone(store.get_snapshot_path(alerts[0].id))
            finally:
                store.close()

    def test_limits_key_frames_to_five(self) -> None:
        capture = _FakeCapture("video.mp4")
        settings = Settings(
            _env_file=None,
            video_sampling_fps=2,
            video_window_size=1,
        )
        with patch("backend.app.video_jobs.cv2.VideoCapture", return_value=capture):
            result = process_video(
                Path("video.mp4"),
                _FakeModelService([0.1] * 6),
                settings,
                lambda progress: self.assertGreaterEqual(progress, 0),
            )
        self.assertEqual(result.sampled_frames, 6)
        self.assertEqual(len(result.key_frames), 5)

    def test_rejects_video_over_duration_limit(self) -> None:
        capture = _FakeCapture("video.mp4")
        settings = Settings(_env_file=None, max_video_duration_seconds=2)
        with patch("backend.app.video_jobs.cv2.VideoCapture", return_value=capture):
            with self.assertRaisesRegex(ValueError, "maximum duration"):
                process_video(
                    Path("video.mp4"),
                    _FakeModelService([]),
                    settings,
                    lambda progress: self.assertGreaterEqual(progress, 0),
                )
        self.assertTrue(capture.released)


class VideoJobManagerTests(unittest.TestCase):
    def test_failed_job_reports_error_and_deletes_temporary_file(self) -> None:
        settings = Settings(_env_file=None)
        model_service = ModelService(settings)
        manager = VideoJobManager(settings, model_service)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "video.mp4"
            path.write_bytes(b"video")
            try:
                with patch(
                    "backend.app.video_jobs.process_video",
                    side_effect=ValueError("Invalid test video."),
                ), patch("backend.app.video_jobs.LOGGER.exception"):
                    submitted = manager.submit(path)
                    deadline = time.monotonic() + 2
                    job = manager.get(submitted.job_id)
                    while job is not None and job.status not in {"failed", "completed"}:
                        if time.monotonic() >= deadline:
                            self.fail("Video job did not finish in time.")
                        time.sleep(0.01)
                        job = manager.get(submitted.job_id)

                self.assertIsNotNone(job)
                self.assertEqual(job.status, "failed")
                self.assertEqual(job.error, "Invalid test video.")
                self.assertFalse(path.exists())
            finally:
                manager.close()


class VideoEndpointTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _load_test_model(service: ModelService) -> None:
        service._model = object()
        service._device = "cpu"
        service._device_name = "cpu"

    async def asyncSetUp(self) -> None:
        self.settings = Settings(_env_file=None, max_video_upload_size_mb=1)
        main.app.dependency_overrides[main.get_settings] = lambda: self.settings
        self.model_load_patch = patch.object(
            ModelService,
            "load",
            self._load_test_model,
        )
        self.model_load_patch.start()
        self.lifespan = main.lifespan(main.app)
        await self.lifespan.__aenter__()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app),
            base_url="http://test",
        )
        login_response = await self.client.post(
            "/auth/login", json={"username": "admin", "password": "admin"}
        )
        self.client.headers.update(
            {"Authorization": f"Bearer {login_response.json()['access_token']}"}
        )

    async def asyncTearDown(self) -> None:
        await self.client.aclose()
        await self.lifespan.__aexit__(None, None, None)
        self.model_load_patch.stop()
        main.app.dependency_overrides.clear()

    async def test_rejects_unsupported_extension_and_oversized_upload(self) -> None:
        unsupported = await self.client.post(
            "/predict-video",
            files={"file": ("sample.mkv", b"video", "video/x-matroska")},
        )
        oversized = await self.client.post(
            "/predict-video",
            files={"file": ("sample.mp4", b"x" * (1024 * 1024 + 1), "video/mp4")},
        )

        self.assertEqual(unsupported.status_code, 415)
        self.assertEqual(oversized.status_code, 413)

    async def test_accepts_video_and_returns_completed_job_results(self) -> None:
        temporary_paths: list[Path] = []

        def finish_job(
            path: Path,
            model_service: ModelService,
            settings: Settings,
            progress_callback,
            *,
            alert_store: AlertStore | None = None,
            source_name: str | None = None,
        ) -> VideoResult:
            self.assertTrue(model_service.is_loaded)
            self.assertGreater(settings.max_video_upload_size_bytes, 0)
            self.assertIsNotNone(alert_store)
            self.assertEqual(source_name, "sample.mp4")
            temporary_paths.append(path)
            self.assertTrue(path.is_file())
            progress_callback(50)
            return _completed_result()

        with patch("backend.app.video_jobs.process_video", side_effect=finish_job):
            response = await self.client.post(
                "/predict-video",
                files={"file": ("sample.mp4", b"video-data", "video/mp4")},
            )
            self.assertEqual(response.status_code, 202, response.text)
            job_id = response.json()["job_id"]

            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                job_response = await self.client.get(f"/jobs/{job_id}")
                if job_response.json()["status"] == "completed":
                    break
                await asyncio.sleep(0.01)

        self.assertEqual(job_response.status_code, 200)
        self.assertEqual(job_response.json()["progress"], 100)
        self.assertEqual(
            job_response.json()["result"]["verdict"],
            "NO_VIOLENCE_ALERT",
        )
        self.assertEqual(len(temporary_paths), 1)
        self.assertFalse(temporary_paths[0].exists())

    async def test_unknown_job_returns_not_found(self) -> None:
        response = await self.client.get("/jobs/missing-job")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
