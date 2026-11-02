import argparse
import json
import sys
import time
from collections import deque

import cv2
import websocket
from websocket import ABNF, WebSocketTimeoutException


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream webcam JPEG frames to the Mob-Alert live WebSocket."
    )
    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:8000/ws/live",
        help="Live WebSocket URL.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV webcam index (default: 0).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=320,
        help="Maximum transmitted frame width (default: 320).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.25,
        help="Seconds between sent frames (default: 0.25).",
    )
    parser.add_argument(
        "--report-interval",
        type=float,
        default=2,
        help="Seconds between FPS/latency reports (default: 2).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Stop after this many seconds and print overall FPS/latency.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Live alert threshold from 0 to 1 (default: 0.8).",
    )
    parser.add_argument(
        "--sensitivity",
        choices=("low", "balanced", "high"),
        default="balanced",
        help="Live alert confirmation sensitivity (default: balanced).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if (
        args.width <= 0
        or args.interval <= 0
        or args.report_interval <= 0
        or (args.duration is not None and args.duration <= 0)
        or not 0 <= args.threshold <= 1
    ):
        print(
            "Width and intervals must be positive, duration must be positive "
            "when provided, and threshold must be between 0 and 1.",
            file=sys.stderr,
        )
        return 2

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        capture.release()
        print(f"Could not open webcam index {args.camera}.", file=sys.stderr)
        return 1

    connection: websocket.WebSocket | None = None
    try:
        connection = websocket.create_connection(args.url, timeout=0.02)
        connection.send(
            json.dumps(
                {
                    "threshold": args.threshold,
                    "sensitivity": args.sensitivity,
                }
            )
        )
        print(f"Streaming webcam {args.camera} to {args.url}; press Ctrl+C to stop.")
        next_frame_at = time.perf_counter()
        stream_started_at = next_frame_at
        report_started_at = next_frame_at
        sent = 0
        received = 0
        latencies: deque[float] = deque(maxlen=200)
        total_sent = 0
        total_received = 0
        total_latency_ms = 0.0
        latency_samples = 0

        while (
            args.duration is None
            or time.perf_counter() - stream_started_at < args.duration
        ):
            now = time.perf_counter()
            if now >= next_frame_at:
                decoded, frame = capture.read()
                if not decoded:
                    raise RuntimeError("Could not read a frame from the webcam.")

                height, width = frame.shape[:2]
                if width > args.width:
                    scaled_height = max(1, round(height * args.width / width))
                    frame = cv2.resize(frame, (args.width, scaled_height))
                encoded, jpeg = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 80],
                )
                if not encoded:
                    raise RuntimeError("Could not encode a webcam frame as JPEG.")
                connection.send(jpeg.tobytes(), opcode=ABNF.OPCODE_BINARY)
                sent += 1
                total_sent += 1
                next_frame_at = now + args.interval

            try:
                message = json.loads(connection.recv())
            except WebSocketTimeoutException:
                continue
            if "error" in message:
                print(f"Server error: {message['error']}", file=sys.stderr)
                continue

            received += 1
            total_received += 1
            inference_ms = message.get("inference_ms")
            if isinstance(inference_ms, (int, float)):
                latencies.append(float(inference_ms))
                total_latency_ms += float(inference_ms)
                latency_samples += 1

            now = time.perf_counter()
            if now - report_started_at >= args.report_interval:
                elapsed = now - report_started_at
                average_latency = (
                    sum(latencies) / len(latencies) if latencies else 0
                )
                print(
                    f"sent_fps={sent / elapsed:.2f} | "
                    f"result_fps={received / elapsed:.2f} | "
                    f"avg_inference_latency={average_latency:.1f} ms | "
                    f"latest_label={message.get('label', 'unknown')} | "
                    f"smoothed_confidence="
                    f"{float(message.get('smoothed_confidence', 0)):.1%} | "
                    f"alert={bool(message.get('alert', False))}"
                )
                report_started_at = now
                sent = 0
                received = 0
        if args.duration is not None:
            elapsed = time.perf_counter() - stream_started_at
            average_latency = (
                total_latency_ms / latency_samples if latency_samples else 0
            )
            print(
                f"summary sent_fps={total_sent / elapsed:.2f} | "
                f"result_fps={total_received / elapsed:.2f} | "
                f"avg_inference_latency={average_latency:.1f} ms | "
                f"latency_samples={latency_samples}"
            )
    except KeyboardInterrupt:
        print("\nWebcam stream stopped.")
    except (OSError, RuntimeError, websocket.WebSocketException, ValueError) as exc:
        print(f"Live webcam test failed: {exc}", file=sys.stderr)
        return 1
    finally:
        capture.release()
        if connection is not None:
            connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
