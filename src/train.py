import argparse
import logging
import shutil
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from src.config import load_config
from src.logging_config import configure_logging


_MODEL_SIZES = ("n", "s", "m")
_SPLITS = ("train", "val", "test")
_CLASSES = ("NonViolence", "Violence")


def _configured_path(config: dict[str, Any], name: str) -> Path:
    path = config["paths"].get(name)
    if not isinstance(path, Path):
        raise ValueError(f"Configured path 'paths.{name}' was not resolved.")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune an Ultralytics YOLO11 classification model."
    )
    parser.add_argument(
        "--model-size",
        "--size",
        choices=_MODEL_SIZES,
        default="n",
        help="YOLO11 classification model size (default: n).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the selected model size's most recent training checkpoint.",
    )
    return parser.parse_args()


def _validate_training_settings(settings: Any) -> dict[str, Any]:
    if not isinstance(settings, dict):
        raise ValueError("Configuration section 'training' must be a mapping.")

    positive_integers = ("epochs", "imgsz", "batch", "patience")
    parsed: dict[str, Any] = {}
    for name in positive_integers:
        value = settings.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Configured training.{name} must be a positive integer.")
        parsed[name] = value

    seed = settings.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Configured training.seed must be an integer.")
    parsed["seed"] = seed

    workers = settings.get("workers")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 0:
        raise ValueError("Configured training.workers must be a non-negative integer.")
    parsed["workers"] = workers

    augment = settings.get("augment")
    if not isinstance(augment, bool):
        raise ValueError("Configured training.augment must be a boolean.")
    parsed["augment"] = augment

    auto_augment = settings.get("auto_augment")
    if augment and (
        not isinstance(auto_augment, str)
        or auto_augment.lower() not in {"randaugment", "augmix", "autoaugment"}
    ):
        raise ValueError(
            "Configured training.auto_augment must be randaugment, augmix, "
            "or autoaugment when augmentation is enabled."
        )
    parsed["auto_augment"] = auto_augment if augment else None

    erasing = settings.get("erasing")
    if isinstance(erasing, bool) or not isinstance(erasing, (int, float)):
        raise ValueError("Configured training.erasing must be numeric.")
    if not 0 <= erasing <= 1:
        raise ValueError("Configured training.erasing must be between 0 and 1.")
    parsed["erasing"] = float(erasing) if augment else 0.0
    return parsed


def _validate_classification_dataset(dataset_dir: Path) -> None:
    for split in _SPLITS:
        for class_name in _CLASSES:
            class_dir = dataset_dir / split / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing classification dataset directory: {class_dir}"
                )
            if not any(class_dir.glob("*.jpg")):
                raise ValueError(f"No .jpg training images found in: {class_dir}")


def _run_training(
    model_size: str,
    resume: bool,
    settings: dict[str, Any],
    dataset_dir: Path,
    run_root: Path,
    best_weights_path: Path,
    logger: logging.Logger,
) -> None:
    run_name = f"yolo11{model_size}_cls"
    run_dir = run_root / run_name
    last_checkpoint = run_dir / "weights" / "last.pt"

    if resume:
        if not last_checkpoint.is_file():
            raise FileNotFoundError(
                f"Cannot resume: checkpoint not found at {last_checkpoint}"
            )
        logger.info("Resuming YOLO11%s classification from %s", model_size, last_checkpoint)
        model = YOLO(str(last_checkpoint))
        model.train(resume=True)
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"Training run already exists: {run_dir}. Use --resume to continue "
                "it, or choose another model size/run directory."
            )
        model = YOLO(f"yolo11{model_size}-cls.pt")
        device = 0 if torch.cuda.is_available() else "cpu"
        logger.info(
            "Starting YOLO11%s classification on %s for up to %d epochs.",
            model_size,
            "CUDA GPU" if device == 0 else "CPU",
            settings["epochs"],
        )
        model.train(
            data=str(dataset_dir),
            epochs=settings["epochs"],
            imgsz=settings["imgsz"],
            batch=settings["batch"],
            patience=settings["patience"],
            seed=settings["seed"],
            deterministic=True,
            workers=settings["workers"],
            device=device,
            auto_augment=settings["auto_augment"],
            erasing=settings["erasing"],
            project=str(run_root),
            name=run_name,
            exist_ok=True,
        )

    run_best_weights = last_checkpoint.parent / "best.pt"
    if not run_best_weights.is_file():
        raise FileNotFoundError(
            f"Training finished without producing best weights: {run_best_weights}"
        )

    best_weights_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_best_weights, best_weights_path)
    logger.info("Best weights saved to %s", best_weights_path)
    print(f"Best weights saved to: {best_weights_path}")


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
    settings = _validate_training_settings(config.get("training"))
    dataset_dir = _configured_path(config, "yolo_cls_dir")
    _validate_classification_dataset(dataset_dir)

    run_root = _configured_path(config, "output_dir") / "classify"
    best_weights_path = _configured_path(config, "model_weights").parent / "best.pt"
    _run_training(
        model_size=args.model_size,
        resume=args.resume,
        settings=settings,
        dataset_dir=dataset_dir,
        run_root=run_root,
        best_weights_path=best_weights_path,
        logger=logger,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
