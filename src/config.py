from pathlib import Path
from typing import Any

import yaml


_REQUIRED_PATHS = (
    "dataset_root",
    "non_violence_dir",
    "violence_dir",
    "output_dir",
    "log_dir",
    "model_weights",
)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration and resolve configured paths from its directory."""
    config_path = config_path.expanduser().resolve()

    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
    except OSError as exc:
        raise OSError(f"Could not read configuration file: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in configuration file: {config_path}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {config_path}")

    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("Configuration section 'paths' must be a mapping.")

    missing_paths = [name for name in _REQUIRED_PATHS if name not in paths]
    if missing_paths:
        raise ValueError(
            "Missing required paths in configuration: " + ", ".join(missing_paths)
        )

    for name, value in paths.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Configured path 'paths.{name}' must be a non-empty string."
            )
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        paths[name] = path.resolve()

    return config
