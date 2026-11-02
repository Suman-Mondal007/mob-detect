import logging
import math
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
from tqdm import tqdm

from src.config import load_config
from src.logging_config import configure_logging


_VIDEO_FORMATS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
}
_REPORT_COLUMNS = (
    "class",
    "video_file",
    "format",
    "is_readable",
    "resolution",
    "width",
    "height",
    "fps",
    "frame_count",
    "duration_seconds",
    "error",
)


def _configured_path(config: dict[str, Any], name: str) -> Path:
    path = config["paths"].get(name)
    if not isinstance(path, Path):
        raise ValueError(f"Configured path 'paths.{name}' was not resolved.")
    return path


def _inspect_video(video_path: Path, class_name: str, dataset_root: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "class": class_name,
        "video_file": video_path.relative_to(dataset_root).as_posix(),
        "format": video_path.suffix.lower(),
        "is_readable": False,
        "resolution": None,
        "width": None,
        "height": None,
        "fps": None,
        "frame_count": None,
        "duration_seconds": None,
        "error": None,
    }

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            row["error"] = "OpenCV could not open the video."
            return row

        decoded, _ = capture.read()
        if not decoded:
            row["error"] = "OpenCV could not decode the first frame."
            return row

        row["is_readable"] = True
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if width > 0 and height > 0:
            row["width"] = width
            row["height"] = height
            row["resolution"] = f"{width}x{height}"

        if math.isfinite(fps) and fps > 0:
            row["fps"] = fps
        if math.isfinite(frame_count) and frame_count > 0:
            row["frame_count"] = int(frame_count)

        if row["fps"] is not None and row["frame_count"] is not None:
            row["duration_seconds"] = row["frame_count"] / row["fps"]
        else:
            row["error"] = "Video opened, but FPS or frame count is unavailable."
    except cv2.error as exc:
        row["error"] = f"OpenCV error: {exc}"
    finally:
        capture.release()

    return row


def _format_stats(values: pd.Series, unit: str = "") -> str:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return "unavailable"
    return (
        f"min={values.min():.2f}{unit}, mean={values.mean():.2f}{unit}, "
        f"median={values.median():.2f}{unit}, max={values.max():.2f}{unit}"
    )


def _print_summary(report: pd.DataFrame, class_names: list[str]) -> None:
    total_videos = len(report)
    readable_videos = int(report["is_readable"].sum()) if total_videos else 0
    unreadable_videos = total_videos - readable_videos
    print("\nDataset inspection summary")
    print(f"Videos scanned: {total_videos}")
    print(f"Readable: {readable_videos}")
    print(f"Unreadable: {unreadable_videos}")

    formats = Counter(report["format"].tolist())
    format_summary = ", ".join(
        f"{suffix}: {count}" for suffix, count in sorted(formats.items())
    )
    print(f"File formats: {format_summary or 'none'}")

    print("\nClass balance:")
    for class_name in class_names:
        class_report = report[report["class"] == class_name]
        class_total = len(class_report)
        percentage = (class_total / total_videos * 100) if total_videos else 0
        class_unreadable = int((~class_report["is_readable"]).sum())
        print(
            f"  {class_name}: {class_total} videos "
            f"({percentage:.1f}%), {class_unreadable} unreadable"
        )

    print("\nVideo metadata by class (readable videos):")
    for class_name in class_names:
        class_report = report[
            (report["class"] == class_name) & report["is_readable"]
        ]
        print(f"  {class_name}:")
        resolutions = class_report["resolution"].value_counts()
        resolution_summary = ", ".join(
            f"{resolution} ({count})"
            for resolution, count in resolutions.head(5).items()
        )
        print(f"    Resolutions: {resolution_summary or 'unavailable'}")
        print(
            "    Width: "
            f"{_format_stats(class_report['width'], ' px')}; "
            "height: "
            f"{_format_stats(class_report['height'], ' px')}"
        )
        print(f"    FPS: {_format_stats(class_report['fps'])}")
        print(
            "    Duration: "
            f"{_format_stats(class_report['duration_seconds'], ' s')}"
        )

    if unreadable_videos:
        print("\nUnreadable videos:")
        for _, row in report.loc[~report["is_readable"], ["video_file", "error"]].iterrows():
            print(f"  {row['video_file']}: {row['error']}")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(project_root / "config.yaml")
    logging_settings = config.get("logging")
    if not isinstance(logging_settings, dict):
        raise ValueError("Configuration section 'logging' must be a mapping.")

    log_level = logging_settings.get("level")
    if not isinstance(log_level, str):
        raise ValueError("Configured logging level must be a string.")

    logger = configure_logging(_configured_path(config, "log_dir"), log_level)
    dataset_root = _configured_path(config, "dataset_root")
    class_directories = (
        ("NonViolence", _configured_path(config, "non_violence_dir")),
        ("Violence", _configured_path(config, "violence_dir")),
    )
    for class_name, class_dir in class_directories:
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"{class_name} dataset directory does not exist: {class_dir}"
            )

    videos = [
        (video_path, class_name)
        for class_name, class_dir in class_directories
        for video_path in sorted(class_dir.iterdir())
        if video_path.is_file() and video_path.suffix.lower() in _VIDEO_FORMATS
    ]
    logger.info("Inspecting %d videos in %s", len(videos), dataset_root)

    records = [
        _inspect_video(video_path, class_name, dataset_root)
        for video_path, class_name in tqdm(videos, desc="Inspecting videos", unit="video")
    ]
    report = pd.DataFrame.from_records(records, columns=_REPORT_COLUMNS)

    data_dir = _configured_path(config, "data_dir")
    data_dir.mkdir(parents=True, exist_ok=True)
    report_path = data_dir / "dataset_report.csv"
    report.to_csv(report_path, index=False, na_rep="")

    _print_summary(report, [name for name, _ in class_directories])
    print(f"\nCSV report saved to: {report_path}")
    logger.info("Dataset report saved to %s", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
