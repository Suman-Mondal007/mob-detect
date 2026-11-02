import argparse
import logging
import math
import sys
import time
from pathlib import Path

import psutil
import torch

from backend.app.camera_manager import CameraManager, camera_worker_source
from backend.app.config import get_settings
from backend.app.model_service import ModelService


LOGGER = logging.getLogger("mob_alert.backend")
VIDEO_SOURCE = Path("demo_videos/sample.mp4")


def _gpu_memory() -> tuple[float, float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / (1024 * 1024)
    reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
    total = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)
    return allocated, reserved, total

def run_benchmark(camera_count: int, duration_seconds: float) -> int:
    settings = get_settings()
    model_service = ModelService(settings)
    camera_manager: CameraManager | None = None
    process = psutil.Process()
    try:
        model_service.load()
        camera_manager = CameraManager(settings, model_service)
        source = str(VIDEO_SOURCE.resolve())
        for index in range(1, camera_count + 1):
            camera_manager.add_camera(
                f"bench-{index:02d}",
                camera_worker_source("file", source),
            )

        print(
            f"Running {camera_count} cameras for {duration_seconds:.0f}s "
            f"on {model_service.device_name}; "
            f"global inference budget={settings.camera_total_inference_fps:g} frames/s."
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        end_at = time.monotonic() + duration_seconds
        while time.monotonic() < end_at:
            time.sleep(min(1, max(0, end_at - time.monotonic())))
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        allocated, reserved, total = _gpu_memory()
        peak_allocated = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if torch.cuda.is_available()
            else 0
        )
        cpu_percent = process.cpu_percent(interval=0.5)
        statuses = {
            status.camera_id: status
            for status in camera_manager.list_cameras()
        }
        avg_input = sum(s.input_fps for s in statuses.values()) / len(statuses)
        avg_infer = sum(s.inference_fps for s in statuses.values()) / len(statuses)
        avg_latency = sum(
            (s.inference_ms or 0) for s in statuses.values()
        ) / len(statuses)
        print(
            f"{'CAMERA':<18} {'FPS IN':>8} {'INFER FPS':>11} "
            f"{'LATENCY MS':>12} {'STATE':>16}"
        )
        for index in range(1, camera_count + 1):
            status = statuses[f"bench-{index:02d}"]
            latency = (
                f"{status.inference_ms:.1f}"
                if status.inference_ms is not None
                else "n/a"
            )
            print(
                f"bench-{index:02d}{'':<10} {status.input_fps:>8.2f} "
                f"{status.inference_fps:>11.2f} {latency:>12} "
                f"{status.state:>16}"
            )
            if status.error:
                print(f"  bench-{index:02d}: {status.error}")

        print(
            f"\nSummary: cameras={camera_count}; avg_input_fps={avg_input:.2f}; "
            f"avg_inference_fps={avg_infer:.2f}; avg_latency_ms={avg_latency:.1f}; "
            f"cpu_percent={cpu_percent:.1f}; "
            f"gpu_allocated={allocated:.0f} MiB; peak={peak_allocated:.0f} MiB; "
            f"gpu_total={total:.0f} MiB."
        )
        return 0
    finally:
        try:
            if camera_manager is not None:
                camera_manager.close()
        finally:
            try:
                model_service.close()
            finally:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure N simulated camera workers and shared inference."
    )
    parser.add_argument("--cameras", type=int, default=6, help="Number of simulated cameras.")
    parser.add_argument("--duration", type=float, default=15, help="Measurement duration in seconds.")
    args = parser.parse_args()
    if args.cameras <= 0:
        parser.error("--cameras must be positive.")
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be a finite positive number.")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        return run_benchmark(args.cameras, args.duration)
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.error("Camera benchmark failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
