import argparse
import csv
import logging
import math
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.config import load_config
from src.logging_config import configure_logging


_VIOLENCE_CLASS = "violence"
_ALERT_CSV_FIELDS = (
    "event_utc",
    "source",
    "video_time_seconds",
    "video_time",
    "smoothed_violence_probability",
    "threshold",
    "alert_duration_seconds",
)


def _configured_path(config: dict[str, Any], name: str) -> Path:
    path = config["paths"].get(name)
    if not isinstance(path, Path):
        raise ValueError(f"Configured path 'paths.{name}' was not resolved.")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run temporally smoothed YOLO11 classification on a video or webcam."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Input video path or webcam index (for example: 0).",
    )
    parser.add_argument("--weights", help="Model checkpoint (default: models/best.pt).")
    parser.add_argument("--output", help="Output video path.")
    parser.add_argument("--alerts-csv", help="Alert log CSV path.")
    parser.add_argument(
        "--threshold",
        type=float,
        help="Smoothed Violence probability threshold (overrides config).",
    )
    parser.add_argument(
        "--alert-seconds",
        type=float,
        help="Seconds the smoothed probability must stay above threshold.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        help="Number of recent frame predictions to average.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open a preview window (useful for headless runs).",
    )
    return parser.parse_args()


def _validate_settings(settings: Any) -> dict[str, int | float | bool]:
    if not isinstance(settings, dict):
        raise ValueError("Configuration section 'inference' must be a mapping.")

    parsed: dict[str, int | float | bool] = {}
    for name in ("imgsz", "window_size"):
        value = settings.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Configured inference.{name} must be a positive integer.")
        parsed[name] = value

    for name in ("violence_threshold", "alert_duration_seconds", "fallback_fps"):
        value = settings.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Configured inference.{name} must be numeric.")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Configured inference.{name} must be finite.")
        parsed[name] = number

    if not 0 <= float(parsed["violence_threshold"]) <= 1:
        raise ValueError("Configured inference.violence_threshold must be between 0 and 1.")
    if float(parsed["alert_duration_seconds"]) < 0:
        raise ValueError("Configured inference.alert_duration_seconds cannot be negative.")
    if float(parsed["fallback_fps"]) <= 0:
        raise ValueError("Configured inference.fallback_fps must be positive.")

    display = settings.get("display")
    if not isinstance(display, bool):
        raise ValueError("Configured inference.display must be a boolean.")
    parsed["display"] = display
    return parsed


def _source_from_arg(source: str) -> int | Path:
    if source.isdecimal():
        return int(source)
    video_path = Path(source).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {video_path}")
    return video_path


def _time_label(seconds: float) -> str:
    total_milliseconds = max(0, int(seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


class _AlertSmoother:
    def __init__(self, window_size: int, threshold: float, duration_seconds: float):
        self.probabilities: deque[float] = deque(maxlen=window_size)
        self.threshold = threshold
        self.duration_seconds = duration_seconds
        self.above_since: float | None = None
        self.alert_active = False

    def update(self, probability: float, timestamp: float) -> tuple[float, bool, bool]:
        self.probabilities.append(probability)
        smoothed_probability = sum(self.probabilities) / len(self.probabilities)

        if smoothed_probability < self.threshold:
            self.above_since = None
            self.alert_active = False
            return smoothed_probability, False, False

        if self.above_since is None:
            self.above_since = timestamp
        alert_started = (
            not self.alert_active
            and timestamp - self.above_since + 1e-9 >= self.duration_seconds
        )
        if alert_started:
            self.alert_active = True
        return smoothed_probability, self.alert_active, alert_started


def _violence_probability(result: Any) -> tuple[str, float, float]:
    if result.probs is None:
        raise RuntimeError(f"No classification probabilities for {result.path}.")

    class_names = result.names
    violence_index = next(
        (
            index
            for index, name in class_names.items()
            if str(name).strip().lower() == _VIOLENCE_CLASS
        ),
        None,
    )
    if violence_index is None:
        raise ValueError(
            "The model checkpoint must contain a class named 'Violence'; "
            f"available classes: {list(class_names.values())}"
        )

    probabilities = result.probs.data.detach().cpu().numpy().reshape(-1)
    if violence_index >= len(probabilities):
        raise RuntimeError("Model class mapping does not match prediction probabilities.")
    top_index = int(np.argmax(probabilities))
    predicted_class = str(class_names[top_index])
    return predicted_class, float(probabilities[top_index]), float(probabilities[violence_index])


def _draw_overlay(
    frame: np.ndarray,
    predicted_class: str,
    confidence: float,
    smoothed_probability: float,
    alert_active: bool,
) -> np.ndarray:
    annotated = frame.copy()
    label = (
        f"{predicted_class}: {confidence:.1%} | "
        f"Smoothed Violence: {smoothed_probability:.1%}"
    )
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 42), (25, 25, 25), -1)
    cv2.putText(
        annotated,
        label,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if alert_active:
        banner_height = min(70, annotated.shape[0])
        cv2.rectangle(
            annotated,
            (0, 42),
            (annotated.shape[1], 42 + banner_height),
            (0, 0, 220),
            -1,
        )
        cv2.putText(
            annotated,
            "ALERT: VIOLENCE DETECTED",
            (10, 42 + min(46, banner_height - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def _write_alert(
    writer: csv.DictWriter,
    alerts_file: Any,
    source_label: str,
    video_seconds: float,
    smoothed_probability: float,
    threshold: float,
    alert_duration_seconds: float,
) -> None:
    writer.writerow(
        {
            "event_utc": datetime.now(timezone.utc).isoformat(),
            "source": source_label,
            "video_time_seconds": f"{video_seconds:.3f}",
            "video_time": _time_label(video_seconds),
            "smoothed_violence_probability": f"{smoothed_probability:.6f}",
            "threshold": f"{threshold:.6f}",
            "alert_duration_seconds": f"{alert_duration_seconds:.3f}",
        }
    )
    alerts_file.flush()


def _run_inference(
    model: YOLO,
    source: int | Path,
    source_label: str,
    output_path: Path,
    alerts_csv_path: Path,
    settings: dict[str, int | float | bool],
    threshold: float,
    alert_duration_seconds: float,
    window_size: int,
    device: int | str,
    logger: logging.Logger,
) -> int:
    if output_path.exists():
        raise FileExistsError(f"Output video already exists: {output_path}")
    if output_path.resolve() == alerts_csv_path.resolve():
        raise ValueError("Output video and alert CSV paths must be different.")

    capture = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open inference source: {source_label}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = float(settings["fallback_fps"])
        logger.warning(
            "Source FPS unavailable; using configured fallback of %.2f FPS.", fps
        )

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Could not determine frame dimensions: {source_label}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    alerts_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if alerts_csv_path.exists() and not alerts_csv_path.is_file():
        capture.release()
        raise OSError(f"Alert CSV path is not a file: {alerts_csv_path}")
    needs_header = not alerts_csv_path.exists() or alerts_csv_path.stat().st_size == 0

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    smoother = _AlertSmoother(window_size, threshold, alert_duration_seconds)
    frame_index = 0
    alert_count = 0
    display = bool(settings["display"])

    try:
        with alerts_csv_path.open("a", newline="", encoding="utf-8") as alerts_file:
            alert_writer = csv.DictWriter(alerts_file, fieldnames=_ALERT_CSV_FIELDS)
            if needs_header:
                alert_writer.writeheader()
                alerts_file.flush()

            while True:
                decoded, frame = capture.read()
                if not decoded:
                    break

                result = model.predict(
                    source=frame,
                    imgsz=int(settings["imgsz"]),
                    device=device,
                    verbose=False,
                )[0]
                predicted_class, confidence, violence_probability = (
                    _violence_probability(result)
                )
                video_seconds = frame_index / fps
                smoothed_probability, alert_active, alert_started = smoother.update(
                    violence_probability, video_seconds
                )

                if alert_started:
                    alert_count += 1
                    _write_alert(
                        alert_writer,
                        alerts_file,
                        source_label,
                        video_seconds,
                        smoothed_probability,
                        threshold,
                        alert_duration_seconds,
                    )
                    logger.warning(
                        "Violence alert at %s (smoothed probability %.1f%%).",
                        _time_label(video_seconds),
                        smoothed_probability * 100,
                    )

                annotated = _draw_overlay(
                    frame,
                    predicted_class,
                    confidence,
                    smoothed_probability,
                    alert_active,
                )
                writer.write(annotated)
                frame_index += 1

                if display:
                    cv2.imshow("Mob-Alert", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("Inference stopped by user.")
                        break
    finally:
        capture.release()
        writer.release()
        if display:
            cv2.destroyAllWindows()

    if frame_index == 0:
        raise RuntimeError(f"No video frames were read from source: {source_label}")

    logger.info(
        "Processed %d frames from %s; saved video to %s; logged %d alert(s) to %s.",
        frame_index,
        source_label,
        output_path,
        alert_count,
        alerts_csv_path,
    )
    print(f"Processed frames: {frame_index}")
    print(f"Output video: {output_path}")
    print(f"Alert log: {alerts_csv_path} ({alert_count} new alert(s))")
    return 0


def main() -> int:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "config.yaml")
    logging_settings = config.get("logging")
    if not isinstance(logging_settings, dict):
        raise ValueError("Configuration section 'logging' must be a mapping.")

    log_level = logging_settings.get("level")
    if not isinstance(log_level, str):
        raise ValueError("Configured logging level must be a string.")

    logger = configure_logging(_configured_path(config, "log_dir"), log_level)
    settings = _validate_settings(config.get("inference"))
    weights_path = (
        Path(args.weights).expanduser().resolve()
        if args.weights
        else _configured_path(config, "model_weights").parent / "best.pt"
    )
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {weights_path}")

    source = _source_from_arg(args.source)
    source_label = str(source)
    inference_dir = _configured_path(config, "output_dir") / "inference"
    default_stem = Path(f"webcam_{source}") if isinstance(source, int) else source
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else inference_dir / f"{default_stem.stem}_annotated.mp4"
    )
    alerts_csv_path = (
        Path(args.alerts_csv).expanduser().resolve()
        if args.alerts_csv
        else inference_dir / "alerts.csv"
    )

    threshold = (
        args.threshold
        if args.threshold is not None
        else float(settings["violence_threshold"])
    )
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1.")

    alert_duration_seconds = (
        args.alert_seconds
        if args.alert_seconds is not None
        else float(settings["alert_duration_seconds"])
    )
    if not math.isfinite(alert_duration_seconds) or alert_duration_seconds < 0:
        raise ValueError("--alert-seconds must be a finite non-negative number.")

    window_size = (
        args.window_size
        if args.window_size is not None
        else int(settings["window_size"])
    )
    if window_size <= 0:
        raise ValueError("--window-size must be a positive integer.")

    if args.no_display:
        settings["display"] = False

    device = 0 if torch.cuda.is_available() else "cpu"
    logger.info(
        "Loading %s for source %s on %s.",
        weights_path,
        source_label,
        "CUDA GPU" if device == 0 else "CPU",
    )
    model = YOLO(str(weights_path))
    return _run_inference(
        model=model,
        source=source,
        source_label=source_label,
        output_path=output_path,
        alerts_csv_path=alerts_csv_path,
        settings=settings,
        threshold=threshold,
        alert_duration_seconds=alert_duration_seconds,
        window_size=window_size,
        device=device,
        logger=logger,
    )


if __name__ == "__main__":
    raise SystemExit(main())
