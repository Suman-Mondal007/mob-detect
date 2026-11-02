import argparse
import csv
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.config import load_config
from src.infer_video import (
    _AlertSmoother,
    _source_from_arg,
    _time_label,
    _validate_settings as _validate_inference_settings,
    _violence_probability,
)
from src.logging_config import configure_logging


_ALERT_CSV_FIELDS = (
    "event_utc",
    "source",
    "video_time_seconds",
    "video_time",
    "smoothed_violence_probability",
    "threshold",
    "alert_duration_seconds",
    "people_detected",
)


def _configured_path(config: dict[str, Any], name: str) -> Path:
    path = config["paths"].get(name)
    if not isinstance(path, Path):
        raise ValueError(f"Configured path 'paths.{name}' was not resolved.")
    return path


def _validate_person_settings(settings: Any) -> dict[str, int | float | str]:
    if not isinstance(settings, dict):
        raise ValueError("Configuration section 'inference.person_detection' must be a mapping.")

    model_name = settings.get("model")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Configured person_detection.model must be a non-empty string.")

    parsed: dict[str, int | float | str] = {"model": model_name}
    for name in ("image_size", "person_class_id"):
        value = settings.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"Configured person_detection.{name} must be an integer."
            )
        if name == "image_size" and value <= 0:
            raise ValueError("Configured person_detection.image_size must be positive.")
        if name == "person_class_id" and value < 0:
            raise ValueError(
                "Configured person_detection.person_class_id cannot be negative."
            )
        parsed[name] = value

    confidence = settings.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("Configured person_detection.confidence must be numeric.")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError(
            "Configured person_detection.confidence must be between 0 and 1."
        )
    parsed["confidence"] = confidence

    context_scale = settings.get("context_scale")
    if isinstance(context_scale, bool) or not isinstance(
        context_scale, (int, float)
    ):
        raise ValueError("Configured person_detection.context_scale must be numeric.")
    context_scale = float(context_scale)
    if not math.isfinite(context_scale) or context_scale < 1:
        raise ValueError(
            "Configured person_detection.context_scale must be finite and at least 1."
        )
    parsed["context_scale"] = context_scale
    return parsed


def _person_regions(
    frame: np.ndarray,
    boxes: np.ndarray,
    context_scale: float,
) -> list[tuple[tuple[int, int, int, int], float, np.ndarray]]:
    height, width = frame.shape[:2]
    regions: list[tuple[tuple[int, int, int, int], float, np.ndarray]] = []
    for raw_x1, raw_y1, raw_x2, raw_y2, confidence in boxes:
        x1, y1, x2, y2 = (
            float(raw_x1),
            float(raw_y1),
            float(raw_x2),
            float(raw_y2),
        )
        person_width = max(1.0, x2 - x1)
        person_height = max(1.0, y2 - y1)
        side = max(person_width, person_height) * context_scale
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        crop_x1 = max(0, int(math.floor(center_x - side / 2)))
        crop_y1 = max(0, int(math.floor(center_y - side / 2)))
        crop_x2 = min(width, int(math.ceil(center_x + side / 2)))
        crop_y2 = min(height, int(math.ceil(center_y + side / 2)))
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            continue

        regions.append(
            (
                (
                    max(0, int(round(x1))),
                    max(0, int(round(y1))),
                    min(width - 1, int(round(x2))),
                    min(height - 1, int(round(y2))),
                ),
                float(confidence),
                frame[crop_y1:crop_y2, crop_x1:crop_x2],
            )
        )
    return regions


def _draw_person_results(
    frame: np.ndarray,
    results: list[tuple[tuple[int, int, int, int], float, float, str, float]],
    smoothed_probability: float,
    alert_active: bool,
) -> np.ndarray:
    annotated = frame.copy()
    header = f"Smoothed Violence: {smoothed_probability:.1%}"
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 42), (25, 25, 25), -1)
    cv2.putText(
        annotated,
        header,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for box, detection_confidence, violence_probability, label, class_confidence in results:
        x1, y1, x2, y2 = box
        color = (
            (0, 0, 220)
            if label.strip().lower() == "violence"
            else (0, 180, 0)
        )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = (
            f"Person {detection_confidence:.0%} | {label} "
            f"{class_confidence:.0%} (V {violence_probability:.0%})"
        )
        text_y = max(62, y1 - 8)
        cv2.putText(
            annotated,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
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


def _classify_person_regions(
    detector: YOLO,
    classifier: YOLO,
    frame: np.ndarray,
    inference_image_size: int,
    detector_settings: dict[str, int | float | str],
    device: int | str,
) -> tuple[
    list[tuple[tuple[int, int, int, int], float, float, str, float]],
    float,
    int,
]:
    detection = detector.predict(
        source=frame,
        imgsz=int(detector_settings["image_size"]),
        conf=float(detector_settings["confidence"]),
        classes=[int(detector_settings["person_class_id"])],
        device=device,
        verbose=False,
    )[0]
    detected_boxes: list[tuple[float, float, float, float, float]] = []
    if detection.boxes is not None and len(detection.boxes) > 0:
        coordinates = detection.boxes.xyxy.detach().cpu().numpy()
        confidences = detection.boxes.conf.detach().cpu().numpy()
        detected_boxes = [
            (
                float(coordinates[index][0]),
                float(coordinates[index][1]),
                float(coordinates[index][2]),
                float(coordinates[index][3]),
                float(confidences[index]),
            )
            for index in range(len(coordinates))
        ]

    regions = _person_regions(
        frame,
        np.asarray(detected_boxes, dtype=np.float32).reshape(-1, 5),
        float(detector_settings["context_scale"]),
    )
    classifications = (
        classifier.predict(
            source=[region for _, _, region in regions],
            imgsz=inference_image_size,
            device=device,
            verbose=False,
        )
        if regions
        else []
    )
    if len(classifications) != len(regions):
        raise RuntimeError(
            "Classifier result count does not match detected person regions."
        )

    overlay_results: list[
        tuple[tuple[int, int, int, int], float, float, str, float]
    ] = []
    frame_violence_probability = 0.0
    for (person_box, detection_confidence, _), classification in zip(
        regions, classifications
    ):
        label, confidence, violence_probability = _violence_probability(
            classification
        )
        frame_violence_probability = max(
            frame_violence_probability, violence_probability
        )
        overlay_results.append(
            (
                person_box,
                detection_confidence,
                violence_probability,
                label,
                confidence,
            )
        )

    return overlay_results, frame_violence_probability, len(regions)


def _run_people_inference(
    detector: YOLO,
    classifier: YOLO,
    source: int | Path,
    source_label: str,
    output_path: Path,
    alerts_csv_path: Path,
    inference_settings: dict[str, int | float | bool],
    detector_settings: dict[str, int | float | str],
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
    if isinstance(source, Path):
        source_path = source.resolve()
        if output_path.resolve() == source_path:
            raise ValueError("Output video path must not overwrite the input video.")
        if alerts_csv_path.resolve() == source_path:
            raise ValueError("Alert CSV path must not overwrite the input video.")

    capture = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open inference source: {source_label}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = float(inference_settings["fallback_fps"])
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
    display = bool(inference_settings["display"])

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

                overlay_results, frame_violence_probability, people_count = (
                    _classify_person_regions(
                        detector,
                        classifier,
                        frame,
                        int(inference_settings["imgsz"]),
                        detector_settings,
                        device,
                    )
                )

                video_seconds = frame_index / fps
                smoothed_probability, alert_active, alert_started = smoother.update(
                    frame_violence_probability, video_seconds
                )
                if alert_started:
                    alert_count += 1
                    alert_writer.writerow(
                        {
                            "event_utc": datetime.now(timezone.utc).isoformat(),
                            "source": source_label,
                            "video_time_seconds": f"{video_seconds:.3f}",
                            "video_time": _time_label(video_seconds),
                            "smoothed_violence_probability": f"{smoothed_probability:.6f}",
                            "threshold": f"{threshold:.6f}",
                            "alert_duration_seconds": f"{alert_duration_seconds:.3f}",
                            "people_detected": len(regions),
                        }
                    )
                    alerts_file.flush()
                    logger.warning(
                        "Violence alert at %s (smoothed probability %.1f%%, "
                        "%d person region(s)).",
                        _time_label(video_seconds),
                        smoothed_probability * 100,
                        people_count,
                    )

                annotated = _draw_person_results(
                    frame, overlay_results, smoothed_probability, alert_active
                )
                writer.write(annotated)
                frame_index += 1
                if display:
                    cv2.imshow("Mob-Alert Person-Crop Inference", annotated)
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect people with YOLO11n and classify person-centered regions "
            "for violence."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Input video path or webcam index (for example: 0).",
    )
    parser.add_argument("--weights", help="Violence classifier checkpoint.")
    parser.add_argument("--detector-weights", help="Person detector weights.")
    parser.add_argument("--output", help="Output video path.")
    parser.add_argument("--alerts-csv", help="Alert log CSV path.")
    parser.add_argument("--threshold", type=float, help="Override Violence threshold.")
    parser.add_argument("--alert-seconds", type=float, help="Override sustained alert duration.")
    parser.add_argument("--window-size", type=int, help="Override smoothing window size.")
    parser.add_argument("--no-display", action="store_true", help="Disable the preview window.")
    return parser.parse_args()


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
    inference_settings = _validate_inference_settings(config.get("inference"))
    detector_settings = _validate_person_settings(
        config.get("inference", {}).get("person_detection")
    )

    classifier_weights = (
        Path(args.weights).expanduser().resolve()
        if args.weights
        else _configured_path(config, "classifier_weights")
    )
    if not classifier_weights.is_file():
        raise FileNotFoundError(
            f"Violence classifier checkpoint does not exist: {classifier_weights}. "
            "Train the classifier first."
        )

    source = _source_from_arg(args.source)
    source_label = str(source)
    inference_dir = _configured_path(config, "output_dir") / "inference" / "people"
    default_stem = Path(f"webcam_{source}") if isinstance(source, int) else source
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else inference_dir / f"{default_stem.stem}_person_cls.mp4"
    )
    alerts_csv_path = (
        Path(args.alerts_csv).expanduser().resolve()
        if args.alerts_csv
        else inference_dir / "alerts.csv"
    )

    threshold = (
        args.threshold
        if args.threshold is not None
        else float(inference_settings["violence_threshold"])
    )
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1.")
    alert_duration_seconds = (
        args.alert_seconds
        if args.alert_seconds is not None
        else float(inference_settings["alert_duration_seconds"])
    )
    if not math.isfinite(alert_duration_seconds) or alert_duration_seconds < 0:
        raise ValueError("--alert-seconds must be a finite non-negative number.")
    window_size = (
        args.window_size
        if args.window_size is not None
        else int(inference_settings["window_size"])
    )
    if window_size <= 0:
        raise ValueError("--window-size must be a positive integer.")
    if args.no_display:
        inference_settings["display"] = False

    device = 0 if torch.cuda.is_available() else "cpu"
    detector_model_name = str(detector_settings["model"])
    detector_weights = (
        Path(args.detector_weights).expanduser().resolve()
        if args.detector_weights
        else Path(detector_model_name).expanduser()
    )
    if args.detector_weights and not detector_weights.is_file():
        raise FileNotFoundError(f"Person detector weights not found: {detector_weights}")
    detector_source = (
        str(detector_weights.resolve())
        if detector_weights.is_file()
        else detector_model_name
    )
    logger.info(
        "Loading person detector %s and violence classifier %s on %s.",
        detector_source,
        classifier_weights,
        "CUDA GPU" if device == 0 else "CPU",
    )
    detector = YOLO(detector_source)
    classifier = YOLO(str(classifier_weights))
    return _run_people_inference(
        detector=detector,
        classifier=classifier,
        source=source,
        source_label=source_label,
        output_path=output_path,
        alerts_csv_path=alerts_csv_path,
        inference_settings=inference_settings,
        detector_settings=detector_settings,
        threshold=threshold,
        alert_duration_seconds=alert_duration_seconds,
        window_size=window_size,
        device=device,
        logger=logger,
    )


if __name__ == "__main__":
    raise SystemExit(main())
