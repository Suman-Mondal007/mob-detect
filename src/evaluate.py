import logging
import random
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm
from ultralytics import YOLO

from src.config import load_config
from src.logging_config import configure_logging


_CLASSES = ("NonViolence", "Violence")
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _configured_path(config: dict[str, Any], name: str) -> Path:
    path = config["paths"].get(name)
    if not isinstance(path, Path):
        raise ValueError(f"Configured path 'paths.{name}' was not resolved.")
    return path


def _validate_settings(settings: Any) -> dict[str, int | float]:
    if not isinstance(settings, dict):
        raise ValueError("Configuration section 'evaluation' must be a mapping.")

    batch_size = settings.get("batch_size")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("Configured evaluation.batch_size must be a positive integer.")

    max_examples = settings.get("max_misclassified_examples")
    if (
        isinstance(max_examples, bool)
        or not isinstance(max_examples, int)
        or max_examples <= 0
    ):
        raise ValueError(
            "Configured evaluation.max_misclassified_examples must be a positive integer."
        )

    recall_threshold = settings.get("violence_recall_warning_threshold")
    if isinstance(recall_threshold, bool) or not isinstance(
        recall_threshold, (int, float)
    ):
        raise ValueError(
            "Configured evaluation.violence_recall_warning_threshold must be numeric."
        )
    recall_threshold = float(recall_threshold)
    if not 0 <= recall_threshold <= 1:
        raise ValueError(
            "Configured evaluation.violence_recall_warning_threshold must be "
            "between 0 and 1."
        )

    return {
        "batch_size": batch_size,
        "max_misclassified_examples": max_examples,
        "violence_recall_warning_threshold": recall_threshold,
    }


def _test_images(dataset_dir: Path) -> list[tuple[Path, str]]:
    images: list[tuple[Path, str]] = []
    for class_name in _CLASSES:
        class_dir = dataset_dir / "test" / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing test class directory: {class_dir}")
        class_images = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not class_images:
            raise ValueError(f"No test images found in: {class_dir}")
        images.extend((image_path, class_name) for image_path in class_images)
    return images


def _predict_test_images(
    model: YOLO,
    images: list[tuple[Path, str]],
    class_names: tuple[str, ...],
    image_size: int,
    batch_size: int,
    device: int | str,
) -> tuple[list[str], list[str]]:
    class_indices = {name: index for index, name in enumerate(class_names)}
    y_true: list[str] = []
    y_pred: list[str] = []

    for start in tqdm(
        range(0, len(images), batch_size),
        desc="Evaluating test images",
        unit="batch",
    ):
        batch = images[start : start + batch_size]
        batch_paths = [path for path, _ in batch]
        results = model.predict(
            source=[str(path) for path in batch_paths],
            imgsz=image_size,
            batch=batch_size,
            device=device,
            verbose=False,
        )
        if len(results) != len(batch):
            raise RuntimeError(
                f"Expected {len(batch)} predictions, received {len(results)}."
            )

        for expected, result in zip(batch, results):
            image_path, true_class = expected
            result_path = Path(result.path).resolve()
            if result_path != image_path.resolve():
                raise RuntimeError(
                    "Prediction order/path mismatch: expected "
                    f"{image_path.resolve()}, received {result_path}."
                )
            if result.probs is None:
                raise RuntimeError(f"No classification probabilities for {result.path}.")

            predicted_index = int(result.probs.top1)
            predicted_class = result.names.get(predicted_index)
            if predicted_class not in class_indices:
                raise ValueError(
                    f"Model returned unexpected class {predicted_class!r} "
                    f"for {result.path}."
                )
            y_true.append(true_class)
            y_pred.append(predicted_class)

    return y_true, y_pred


def _save_confusion_matrix(
    matrix: np.ndarray,
    class_names: tuple[str, ...],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Test split confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _save_misclassified_examples(
    images: list[tuple[Path, str]],
    y_true: list[str],
    y_pred: list[str],
    output_dir: Path,
    limit: int,
    seed: int,
) -> int:
    mistakes = [
        (image_path, true_class, predicted_class)
        for (image_path, true_class), predicted_class in zip(images, y_pred)
        if true_class != predicted_class
    ]
    selected = random.Random(seed).sample(mistakes, min(limit, len(mistakes)))

    for old_example in output_dir.glob("misclassified_*.jpg"):
        old_example.unlink()

    for index, (image_path, true_class, predicted_class) in enumerate(
        selected, start=1
    ):
        image = cv2.imread(str(image_path))
        if image is None:
            raise OSError(f"Could not read misclassified test image: {image_path}")

        banner_height = max(34, image.shape[0] // 8)
        banner = np.zeros((banner_height, image.shape[1], 3), dtype=np.uint8)
        banner[:] = (30, 30, 30)
        text = f"True: {true_class} | Predicted: {predicted_class}"
        cv2.putText(
            banner,
            text,
            (8, min(banner_height - 8, 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        annotated = np.vstack((banner, image))
        output_path = output_dir / (
            f"misclassified_{index:02d}_true-{true_class}"
            f"_pred-{predicted_class}_{image_path.stem}.jpg"
        )
        if not cv2.imwrite(str(output_path), annotated):
            raise OSError(f"Could not save misclassified example: {output_path}")

    return len(selected)


def _print_metrics(
    y_true: list[str],
    y_pred: list[str],
    class_names: tuple[str, ...],
    violence_recall_threshold: float,
) -> np.ndarray:
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(class_names),
        zero_division=0,
    )
    print("\nTest split metrics")
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    for index, class_name in enumerate(class_names):
        print(
            f"{class_name}: precision={precision[index]:.4f}, "
            f"recall={recall[index]:.4f}, f1={f1[index]:.4f}, "
            f"support={support[index]}"
        )

    if recall[class_names.index("Violence")] < violence_recall_threshold:
        print(
            "\nWARNING: Violence recall is below the configured "
            f"{violence_recall_threshold:.0%} threshold. Review the false negatives "
            "and consider collecting more varied Violence videos, improving "
            "video-level split coverage, tuning the training setup, or evaluating "
            "a more suitable temporal action-recognition model."
        )

    return confusion_matrix(y_true, y_pred, labels=list(class_names))


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
    settings = _validate_settings(config.get("evaluation"))
    weights_path = _configured_path(config, "model_weights").parent / "best.pt"
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Trained model weights were not found: {weights_path}. "
            "Complete training with src.train first."
        )

    dataset_dir = _configured_path(config, "yolo_cls_dir")
    images = _test_images(dataset_dir)
    image_size = config.get("training", {}).get("imgsz")
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("Configured training.imgsz must be a positive integer.")

    device = 0 if torch.cuda.is_available() else "cpu"
    logger.info(
        "Evaluating %d test images with %s on %s.",
        len(images),
        weights_path,
        "CUDA GPU" if device == 0 else "CPU",
    )
    model = YOLO(str(weights_path))
    y_true, y_pred = _predict_test_images(
        model=model,
        images=images,
        class_names=_CLASSES,
        image_size=image_size,
        batch_size=int(settings["batch_size"]),
        device=device,
    )

    output_dir = _configured_path(config, "output_dir") / "errors"
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "confusion_matrix.png"
    matrix = _print_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=_CLASSES,
        violence_recall_threshold=float(
            settings["violence_recall_warning_threshold"]
        ),
    )
    _save_confusion_matrix(matrix, _CLASSES, matrix_path)
    examples_saved = _save_misclassified_examples(
        images=images,
        y_true=y_true,
        y_pred=y_pred,
        output_dir=output_dir,
        limit=int(settings["max_misclassified_examples"]),
        seed=int(config.get("data_preparation", {}).get("random_seed", 42)),
    )

    errors = sum(true_class != predicted_class for true_class, predicted_class in zip(y_true, y_pred))
    print(f"\nMisclassified examples saved: {examples_saved} (out of {errors} errors)")
    print(f"Confusion matrix saved to: {matrix_path}")
    print(f"Misclassified images saved under: {output_dir}")
    logger.info(
        "Evaluation complete on the test split only; confusion matrix: %s; "
        "misclassified examples saved: %d.",
        matrix_path,
        examples_saved,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
