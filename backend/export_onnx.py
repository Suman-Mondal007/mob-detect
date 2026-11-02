import sys
from pathlib import Path

from backend.app.config import get_settings
from backend.app.model_service import ModelService


def main() -> int:
    settings = get_settings()
    service = ModelService(settings)
    service.load()
    try:
        exported = service._model.export(format="onnx")
        print(f"Exported ONNX model to: {exported}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ONNX export failed: {exc}")
        return 1
    finally:
        service.close()


if __name__ == "__main__":
    sys.exit(main())
