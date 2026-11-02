import argparse
import json
import mimetypes
import sys
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _request_json(request: Request, timeout: float) -> dict[str, object]:
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach the API: {exc.reason}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("The API returned an unexpected JSON response.")
    return payload


def _check_api(base_url: str, image_path: Path, timeout: float) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    health = _request_json(Request(f"{base_url}/health"), timeout)
    if health.get("model_loaded") is not True:
        raise RuntimeError(f"The model is not loaded: {health}")
    print(
        "Health check passed "
        f"(device={health.get('device')}, version={health.get('version')})."
    )

    content_type = mimetypes.guess_type(image_path.name)[0]
    if content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError("Choose a JPEG, PNG, or WebP image for the smoke test.")

    boundary = uuid.uuid4().hex
    file_bytes = image_path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{image_path.name}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("ascii")
    request = Request(
        f"{base_url}/predict-image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    prediction = _request_json(request, timeout)

    predicted_class = prediction.get("predicted_class")
    confidence = prediction.get("confidence")
    violence_probability = prediction.get("violence_probability")
    if (
        not isinstance(predicted_class, str)
        or not isinstance(confidence, (int, float))
        or not isinstance(violence_probability, (int, float))
        or not 0 <= confidence <= 1
        or not 0 <= violence_probability <= 1
    ):
        raise RuntimeError(f"The prediction response is invalid: {prediction}")

    print(
        "Image prediction passed "
        f"(class={predicted_class}, confidence={confidence:.1%}, "
        f"violence_probability={violence_probability:.1%})."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test the running Mob-Alert API with an image."
    )
    parser.add_argument("image", type=Path, help="JPEG, PNG, or WebP image to upload.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="API base URL (default: http://127.0.0.1:8000).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=90,
        help="HTTP timeout in seconds (default: 90).",
    )
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout must be positive.")
    try:
        _check_api(args.base_url.rstrip("/"), args.image.expanduser(), args.timeout)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"API smoke test failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
