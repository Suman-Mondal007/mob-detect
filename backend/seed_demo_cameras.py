import logging
import sys
from pathlib import Path

from backend.app.camera_store import Camera, CameraStore
from backend.app.config import get_settings


LOGGER = logging.getLogger("mob_alert.backend")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def seed_demo_cameras(
    video_directory: Path,
    camera_store: CameraStore,
) -> int:
    videos = sorted(
        path.resolve()
        for path in video_directory.iterdir()
        if path.is_file() and path.suffix.lower() in DEMO_VIDEO_EXTENSIONS
    ) if video_directory.is_dir() else []
    if not videos:
        raise ValueError(
            f"Expected at least one supported video file in {video_directory}."
        )

    for index in range(1, 7):
        video_path = videos[(index - 1) % len(videos)]
        camera_id = f"demo-{index:02d}"
        values = {
            "name": f"Demo Camera {index}",
            "location": f"Simulation Zone {index}",
            "source_type": "file",
            "source_uri": str(video_path),
            "enabled": True,
        }
        existing = camera_store.get(camera_id)
        if existing is None:
            camera_store.create(Camera(id=camera_id, **values))
        else:
            camera_store.update(camera_id, **values)
        LOGGER.info("Seeded %s from %s.", camera_id, video_path.name)
    return 6


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    settings = get_settings()
    store = CameraStore(settings.resolved_alerts_database_path)
    try:
        count = seed_demo_cameras(PROJECT_ROOT / "demo_videos", store)
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.error("Could not seed demo cameras: %s", exc)
        return 1
    finally:
        store.close()
    print(f"Seeded {count} demo cameras. Start the backend to open their video sources.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
