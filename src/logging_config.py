import logging
from pathlib import Path


def configure_logging(log_dir: Path, level_name: str) -> logging.Logger:
    """Configure console and file logging for the application."""
    normalized_level = level_name.upper()
    level = logging.getLevelName(normalized_level)
    if not isinstance(level, int):
        raise ValueError(f"Unsupported logging level: {level_name}")

    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mob_alert")
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        file_handler = logging.FileHandler(log_dir / "mob_alert.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
