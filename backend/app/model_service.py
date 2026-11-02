import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from typing import Any, Callable, TypeVar

import numpy as np
import torch
from ultralytics import YOLO

from backend.app.config import Settings
from backend.app.schemas import PredictionResponse
from src.infer_video import _violence_probability


LOGGER = logging.getLogger("mob_alert.backend")
ResultT = TypeVar("ResultT")


class InferenceBusyError(RuntimeError):
    """Raised when an earlier timed-out inference is still running."""


class InferenceTimeoutError(TimeoutError):
    """Raised when an image prediction exceeds its configured time limit."""


class ModelService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: YOLO | None = None
        self._device: int | str | None = None
        self._device_name = "unavailable"
        self._executor: ThreadPoolExecutor | None = None
        self._active_prediction: Future[Any] | None = None
        self._condition = threading.Condition()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device_name(self) -> str:
        return self._device_name

    def load(self) -> None:
        if self._model is not None:
            raise RuntimeError("The model service has already been loaded.")

        model_path = self._settings.resolved_model_path
        if not model_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        self._device = 0 if torch.cuda.is_available() else "cpu"
        self._device_name = "cuda:0" if self._device == 0 else "cpu"
        LOGGER.info("Loading classifier from %s on %s.", model_path, self._device_name)
        self._model = YOLO(str(model_path))
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="mob-alert-inference",
        )
        LOGGER.info("Classifier loaded on %s.", self._device_name)

    def predict(
        self,
        image: np.ndarray,
        *,
        wait_for_available: bool = False,
    ) -> PredictionResponse:
        return self._run_prediction(
            lambda: self._predict(image),
            wait_for_available=wait_for_available,
        )

    def predict_batch(
        self,
        images: list[np.ndarray],
        *,
        wait_for_available: bool = False,
    ) -> list[PredictionResponse]:
        if not images:
            return []
        return self._run_prediction(
            lambda: self._predict_batch(images),
            wait_for_available=wait_for_available,
        )

    def _run_prediction(
        self,
        prediction: Callable[[], ResultT],
        *,
        wait_for_available: bool,
    ) -> ResultT:
        if self._model is None or self._executor is None or self._device is None:
            raise RuntimeError("The classifier is not loaded.")

        deadline = time.monotonic() + self._settings.predict_timeout_seconds
        with self._condition:
            active = self._active_prediction
            while active is not None and not active.done():
                if not wait_for_available:
                    raise InferenceBusyError(
                        "A previous image prediction is still running; try again shortly."
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise InferenceTimeoutError(
                        "Timed out waiting for another image prediction to finish."
                    )
                self._condition.wait(timeout=remaining)
                active = self._active_prediction

            future = self._executor.submit(prediction)
            self._active_prediction = future

        future.add_done_callback(self._clear_active_prediction)
        try:
            return future.result(timeout=max(0, deadline - time.monotonic()))
        except TimeoutError as exc:
            raise InferenceTimeoutError(
                "Image prediction exceeded the configured time limit."
            ) from exc

    def close(self) -> None:
        executor = self._executor
        self._executor = None
        self._model = None
        self._device = None
        self._device_name = "unavailable"
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def _predict(self, image: np.ndarray) -> PredictionResponse:
        results = self._predict_source(image)
        if not results:
            raise RuntimeError("The classifier returned no prediction.")
        return self._prediction_response(results[0])

    def _predict_batch(self, images: list[np.ndarray]) -> list[PredictionResponse]:
        results = self._predict_source(images)
        if len(results) != len(images):
            raise RuntimeError(
                "The classifier returned a different number of predictions than frames."
            )
        return [self._prediction_response(result) for result in results]

    def _predict_source(self, source: np.ndarray | list[np.ndarray]) -> list[Any]:
        if self._model is None or self._device is None:
            raise RuntimeError("The classifier is not loaded.")

        try:
            results = self._predict_on_device(source, self._device)
        except (OSError, RuntimeError):
            if self._device != 0:
                raise
            LOGGER.warning(
                "CUDA inference failed; retrying this and future predictions on CPU.",
                exc_info=True,
            )
            self._device = "cpu"
            self._device_name = "cpu"
            results = self._predict_on_device(source, self._device)

        return results

    @staticmethod
    def _prediction_response(result: Any) -> PredictionResponse:
        predicted_class, confidence, violence_probability = _violence_probability(
            result
        )
        return PredictionResponse(
            predicted_class=predicted_class,
            confidence=confidence,
            violence_probability=violence_probability,
        )

    def _predict_on_device(
        self,
        source: np.ndarray | list[np.ndarray],
        device: int | str,
    ) -> list[Any]:
        if self._model is None:
            raise RuntimeError("The classifier is not loaded.")

        return self._model.predict(
            source=source,
            imgsz=self._settings.image_size,
            device=device,
            quantize="fp16" if device == 0 else None,
            verbose=False,
        )

    def _clear_active_prediction(self, future: Future[Any]) -> None:
        with self._condition:
            if self._active_prediction is future:
                self._active_prediction = None
            self._condition.notify_all()
