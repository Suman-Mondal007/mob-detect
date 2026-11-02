import sys

import torch
import ultralytics


def main() -> int:
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Ultralytics version: {ultralytics.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA/GPU available: {'Yes' if cuda_available else 'No'}")

    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(
            "No GPU is available to PyTorch. For GPU-accelerated training, "
            "consider Google Colab or Kaggle."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
