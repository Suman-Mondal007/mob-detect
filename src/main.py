from pathlib import Path
from typing import Any

from src.config import load_config
from src.logging_config import configure_logging


def _configured_path(config: dict[str, Any], name: str) -> Path:
    paths = config["paths"]
    path = paths[name]
    if not isinstance(path, Path):
        raise ValueError(f"Configured path 'paths.{name}' was not resolved.")
    return path


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config.yaml"
    config = load_config(config_path)
    logging_settings = config.get("logging")
    if not isinstance(logging_settings, dict):
        raise ValueError("Configuration section 'logging' must be a mapping.")

    log_level = logging_settings.get("level")
    if not isinstance(log_level, str):
        raise ValueError("Configured logging level must be a string.")

    logger = configure_logging(_configured_path(config, "log_dir"), log_level)
    logger.info("Starting Mob-Alert project setup validation.")

    required_directories = (
        ("dataset_root", "Dataset root"),
        ("non_violence_dir", "NonViolence class directory"),
        ("violence_dir", "Violence class directory"),
    )
    for path_name, description in required_directories:
        path = _configured_path(config, path_name)
        if not path.is_dir():
            logger.error("%s does not exist or is not a directory: %s", description, path)
            raise FileNotFoundError(f"{description} does not exist: {path}")

    output_dir = _configured_path(config, "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Project setup validated.")
    logger.info("Dataset root: %s", _configured_path(config, "dataset_root"))
    logger.info(
        "Class directories: %s, %s",
        _configured_path(config, "non_violence_dir").name,
        _configured_path(config, "violence_dir").name,
    )
    logger.info("Output directory: %s", output_dir)
    logger.info("Log file: %s", _configured_path(config, "log_dir") / "mob_alert.log")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
