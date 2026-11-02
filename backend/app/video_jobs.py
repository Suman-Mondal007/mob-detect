import base64
import logging
import math
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from backend.app.alerts_store import AlertStore
from backend.app.config import Settings
from backend.app.model_service import ModelService
from backend.app.schemas import (
    AlertRange,
    KeyFrame,
    TimelinePoint,
    VideoJobResponse,
    VideoResult,
)
from src.infer_video import _AlertSmoother


LOGGER = logging.getLogger("mob_alert.backend")
KEY_FRAME_SIZE = 480
KEY_FRAME_JPEG_QUALITY = 75


@dataclass
class _VideoJob:
    job_id: str
    path: Path
    source_name: str
    status: str = "queued"
    progress: float = 0
    result: VideoResult | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    future: Future[None] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> VideoJobResponse:
        with self.lock:
            return VideoJobResponse(
                job_id=self.job_id,
                status=self.status,
                progress=self.progress,
                result=self.result,
                error=self.error,
            )

    def update_progress(self, progress: float) -> None:
        with self.lock:
            self.progress = min(99.0, max(self.progress, progress))
            if self.status == "queued":
                self.status = "processing"


class VideoJobManager:
    def __init__(
        self,
        settings: Settings,
        model_service: ModelService,
        alert_store: AlertStore | None = None,
    ) -> None:
        self._settings = settings
        self._model_service = model_service
        self._alert_store = alert_store
        self._jobs: dict[str, _VideoJob] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mob-alert-video",
        )

    def submit(self, path: Path, source_name: str | None = None) -> VideoJobResponse:
        with self._lock:
            self._prune_completed_jobs()
            active_count = sum(
                self._job_status(job) in {"queued", "processing"}
                for job in self._jobs.values()
            )
            if active_count >= self._settings.max_active_video_jobs:
                raise RuntimeError("The video-processing queue is full; try again later.")
            if len(self._jobs) >= self._settings.max_stored_video_jobs:
                raise RuntimeError("Video job history is full; try again later.")

            job = _VideoJob(
                job_id=uuid.uuid4().hex,
                path=path,
                source_name=source_name or path.name,
            )
            self._jobs[job.job_id] = job
            job.future = self._executor.submit(self._run_job, job)
            return job.snapshot()

    def get(self, job_id: str) -> VideoJobResponse | None:
        with self._lock:
            job = self._jobs.get(job_id)
        return job.snapshot() if job is not None else None

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.future is not None and job.future.cancelled():
                try:
                    job.path.unlink(missing_ok=True)
                except OSError:
                    LOGGER.exception(
                        "Could not remove temporary video for cancelled job %s.",
                        job.job_id,
                    )

    def _prune_completed_jobs(self) -> None:
        completed = sorted(
            (
                job
                for job in self._jobs.values()
                if self._job_status(job) in {"completed", "failed"}
            ),
            key=lambda job: job.created_at,
        )
        excess = len(self._jobs) - self._settings.max_stored_video_jobs + 1
        for job in completed[:max(0, excess)]:
            self._jobs.pop(job.job_id, None)

    @staticmethod
    def _job_status(job: _VideoJob) -> str:
        with job.lock:
            return job.status

    def _run_job(self, job: _VideoJob) -> None:
        result: VideoResult | None = None
        error: str | None = None
        try:
            result = process_video(
                job.path,
                self._model_service,
                self._settings,
                job.update_progress,
                alert_store=self._alert_store,
                source_name=job.source_name,
            )
        except Exception as exc:
            LOGGER.exception("Video processing failed for job %s.", job.job_id)
            error = str(exc) or "Video processing failed."
        finally:
            try:
                job.path.unlink(missing_ok=True)
            except OSError as exc:
                LOGGER.exception(
                    "Could not remove temporary video for job %s.",
                    job.job_id,
                )
                result = None
                error = f"Could not delete the temporary video: {exc}"

            with job.lock:
                job.result = result
                job.error = error
                if error is None:
                    job.progress = 100
                    job.status = "completed"
                else:
                    job.status = "failed"


def _encode_key_frame(frame: np.ndarray) -> str:
    height, width = frame.shape[:2]
    longest_side = max(width, height)
    if longest_side > KEY_FRAME_SIZE:
        scale = KEY_FRAME_SIZE / longest_side
        frame = cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
        )
    encoded, buffer = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, KEY_FRAME_JPEG_QUALITY],
    )
    if not encoded:
        raise RuntimeError("Could not encode a video key frame.")
    return base64.b64encode(buffer).decode("ascii")


def process_video(
    video_path: Path,
    model_service: ModelService,
    settings: Settings,
    progress_callback: Callable[[float], None],
    *,
    alert_store: AlertStore | None = None,
    source_name: str | None = None,
) -> VideoResult:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("The uploaded file is not a readable video.")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        capture.release()
        raise ValueError("Could not determine the uploaded video's duration or frame rate.")

    duration = frame_count / fps
    if duration > settings.max_video_duration_seconds:
        capture.release()
        raise ValueError(
            "Video exceeds the maximum duration of "
            f"{settings.max_video_duration_seconds} seconds."
        )
    if width <= 0 or height <= 0 or width * height > settings.max_video_frame_pixels:
        capture.release()
        raise ValueError("Video frame dimensions exceed the configured pixel limit.")

    frame_interval = max(1, round(fps / settings.video_sampling_fps))
    smoother = _AlertSmoother(
        settings.video_window_size,
        settings.violence_threshold,
        settings.video_alert_duration_seconds,
    )
    timeline: list[TimelinePoint] = []
    alert_ranges: list[AlertRange] = []
    key_frames: list[tuple[float, KeyFrame]] = []
    maximum_probability = 0.0
    active_alert_start: float | None = None
    frame_index = 0

    try:
        while True:
            decoded, frame = capture.read()
            if not decoded:
                break
            if frame is None:
                raise ValueError("Could not decode a frame from the uploaded video.")
            actual_height, actual_width = frame.shape[:2]
            if actual_width * actual_height > settings.max_video_frame_pixels:
                raise ValueError(
                    "Video frame dimensions exceed the configured pixel limit."
                )

            if frame_index % frame_interval == 0:
                timestamp = frame_index / fps
                prediction = model_service.predict(frame, wait_for_available=True)
                probability = prediction.violence_probability
                maximum_probability = max(maximum_probability, probability)
                smoothed, alert_active, alert_started = smoother.update(
                    probability,
                    timestamp,
                )
                timeline.append(
                    TimelinePoint(
                        time_seconds=timestamp,
                        violence_probability=smoothed,
                    )
                )
                if alert_active and active_alert_start is None:
                    active_alert_start = timestamp
                if alert_started and alert_store is not None:
                    alert_store.record(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        source="upload",
                        source_name=source_name or video_path.name,
                        label=prediction.predicted_class,
                        confidence=prediction.confidence,
                        smoothed_confidence=smoothed,
                        threshold=settings.violence_threshold,
                        snapshot_frame=frame,
                        video_time_seconds=timestamp,
                    )
                if not alert_active and active_alert_start is not None:
                    alert_ranges.append(
                        AlertRange(
                            start_seconds=active_alert_start,
                            end_seconds=timestamp,
                        )
                    )
                    active_alert_start = None

                if len(key_frames) < 5 or smoothed > key_frames[-1][0]:
                    key_frame = KeyFrame(
                        time_seconds=timestamp,
                        violence_probability=smoothed,
                        image_base64=_encode_key_frame(frame),
                    )
                    key_frames.append((smoothed, key_frame))
                    key_frames.sort(
                        key=lambda item: (-item[0], item[1].time_seconds)
                    )
                    del key_frames[5:]

            frame_index += 1
            progress_callback(frame_index / frame_count * 100)

        if not timeline:
            raise ValueError("The uploaded video contains no decodable frames.")
    finally:
        capture.release()

    if active_alert_start is not None:
        alert_ranges.append(
            AlertRange(start_seconds=active_alert_start, end_seconds=duration)
        )

    return VideoResult(
        verdict="VIOLENCE_ALERT" if alert_ranges else "NO_VIOLENCE_ALERT",
        max_violence_probability=maximum_probability,
        sampled_frames=len(timeline),
        timeline=timeline,
        alert_ranges=alert_ranges,
        key_frames=[item[1] for item in key_frames],
    )
