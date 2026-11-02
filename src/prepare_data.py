import hashlib
import logging
import math
from pathlib import Path
from typing import Any

import cv2
from sklearn.model_selection import train_test_split
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
_SPLITS = ("train", "val", "test")
_CLASSES = ("NonViolence", "Violence")


def _configured_path(config: dict[str, Any], name: str) -> Path:
    path = config["paths"].get(name)
    if not isinstance(path, Path):
        raise ValueError(f"Configured path 'paths.{name}' was not resolved.")
    return path


def _validate_settings(settings: Any) -> dict[str, int | float]:
    if not isinstance(settings, dict):
        raise ValueError("Configuration section 'data_preparation' must be a mapping.")

    integer_settings = (
        "frame_interval",
        "max_frames_per_video",
        "image_size",
        "random_seed",
    )
    parsed: dict[str, int | float] = {}
    for name in integer_settings:
        value = settings.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Configured data_preparation.{name} must be an integer.")
        if name != "random_seed" and value <= 0:
            raise ValueError(f"Configured data_preparation.{name} must be positive.")
        parsed[name] = value

    ratios = []
    for name in ("train_ratio", "val_ratio", "test_ratio"):
        value = settings.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Configured data_preparation.{name} must be numeric.")
        ratio = float(value)
        if not math.isfinite(ratio) or not 0 < ratio < 1:
            raise ValueError(
                f"Configured data_preparation.{name} must be between 0 and 1."
            )
        parsed[name] = ratio
        ratios.append(ratio)

    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Train, validation, and test ratios must sum to 1.")

    return parsed


def _split_videos(
    videos: list[tuple[Path, str]],
    ratios: dict[str, int | float],
    seed: int,
) -> dict[str, list[tuple[Path, str]]]:
    if not videos:
        raise ValueError("No supported video files were found in the class folders.")

    paths = [video_path for video_path, _ in videos]
    labels = [class_name for _, class_name in videos]
    train_ratio = float(ratios["train_ratio"])
    val_ratio = float(ratios["val_ratio"])
    test_ratio = float(ratios["test_ratio"])

    try:
        train_paths, remaining_paths, train_labels, remaining_labels = train_test_split(
            paths,
            labels,
            train_size=train_ratio,
            random_state=seed,
            stratify=labels,
        )
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            remaining_paths,
            remaining_labels,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=seed,
            stratify=remaining_labels,
        )
    except ValueError as exc:
        raise ValueError(
            "Could not create a stratified video-level train/val/test split. "
            "Check that each class contains enough videos and that split ratios "
            "are valid."
        ) from exc

    def pair(
        paths_in_split: list[Path], labels_in_split: list[str]
    ) -> list[tuple[Path, str]]:
        return list(zip(paths_in_split, labels_in_split))

    return {
        "train": pair(train_paths, train_labels),
        "val": pair(val_paths, val_labels),
        "test": pair(test_paths, test_labels),
    }


def _prepare_output_directories(output_root: Path) -> None:
    for split in _SPLITS:
        for class_name in _CLASSES:
            class_output_dir = output_root / split / class_name
            class_output_dir.mkdir(parents=True, exist_ok=True)
            for existing_image in class_output_dir.glob("*.jpg"):
                existing_image.unlink()


def _extract_video_frames(
    video_path: Path,
    class_name: str,
    dataset_root: Path,
    output_dir: Path,
    frame_interval: int,
    max_frames: int,
    image_size: int,
) -> tuple[int, str | None]:
    relative_video_path = video_path.relative_to(dataset_root).as_posix()
    video_id = hashlib.sha256(relative_video_path.encode("utf-8")).hexdigest()[:12]
    capture = cv2.VideoCapture(str(video_path))
    frames_written = 0
    frame_index = 0
    error: str | None = None

    try:
        if not capture.isOpened():
            return 0, "OpenCV could not open the video."

        while frames_written < max_frames:
            decoded, frame = capture.read()
            if not decoded:
                if frames_written == 0:
                    error = "OpenCV could not decode a frame."
                break

            if frame_index % frame_interval == 0:
                resized_frame = cv2.resize(
                    frame,
                    (image_size, image_size),
                    interpolation=cv2.INTER_AREA,
                )
                image_path = (
                    output_dir
                    / f"{video_path.stem}__{video_id}__{frames_written:04d}.jpg"
                )
                if not cv2.imwrite(str(image_path), resized_frame):
                    raise OSError(f"OpenCV could not write image: {image_path}")
                frames_written += 1

            frame_index += 1
    except cv2.error as exc:
        error = f"OpenCV error: {exc}"
    finally:
        capture.release()

    if frames_written == 0 and error is None:
        error = "No frames were extracted."
    return frames_written, error


def _print_summary(
    splits: dict[str, list[tuple[Path, str]]],
    image_counts: dict[str, dict[str, int]],
    failures: int,
) -> None:
    print("\nFrame preparation summary")
    total_videos = 0
    total_images = 0
    for split in _SPLITS:
        split_videos = len(splits[split])
        split_images = sum(image_counts[split].values())
        total_videos += split_videos
        total_images += split_images
        print(f"{split}: {split_videos} videos")
        for class_name in _CLASSES:
            print(f"  {class_name}: {image_counts[split][class_name]} images")
        print(f"  Total images: {split_images}")
    print(f"Videos assigned: {total_videos}")
    print(f"Total images written: {total_images}")
    print(f"Videos with extraction errors: {failures}")


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
    settings = _validate_settings(config.get("data_preparation"))
    dataset_root = _configured_path(config, "dataset_root")
    class_directories = {
        "NonViolence": _configured_path(config, "non_violence_dir"),
        "Violence": _configured_path(config, "violence_dir"),
    }
    for class_name, class_dir in class_directories.items():
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"{class_name} dataset directory does not exist: {class_dir}"
            )

    videos = [
        (video_path, class_name)
        for class_name, class_dir in class_directories.items()
        for video_path in sorted(class_dir.iterdir())
        if video_path.is_file() and video_path.suffix.lower() in _VIDEO_FORMATS
    ]
    seed = int(settings["random_seed"])
    splits = _split_videos(videos, settings, seed)

    output_root = _configured_path(config, "yolo_cls_dir")
    _prepare_output_directories(output_root)
    image_counts = {
        split: {class_name: 0 for class_name in _CLASSES} for split in _SPLITS
    }
    failures = 0

    for split in _SPLITS:
        for video_path, class_name in tqdm(
            splits[split], desc=f"Extracting {split}", unit="video"
        ):
            frames_written, error = _extract_video_frames(
                video_path=video_path,
                class_name=class_name,
                dataset_root=dataset_root,
                output_dir=output_root / split / class_name,
                frame_interval=int(settings["frame_interval"]),
                max_frames=int(settings["max_frames_per_video"]),
                image_size=int(settings["image_size"]),
            )
            image_counts[split][class_name] += frames_written
            if error is not None:
                failures += 1
                logger.error("%s: %s", video_path, error)

    _print_summary(splits, image_counts, failures)
    print(f"Classification dataset saved to: {output_root}")
    logger.info("Prepared classification dataset at %s", output_root)
    if failures:
        logger.error(
            "Frame preparation completed with %d video extraction error(s).",
            failures,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
