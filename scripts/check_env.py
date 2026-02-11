#!/usr/bin/env python3
"""Quick compatibility checks for local training environment."""

from __future__ import annotations

import importlib
import platform
import sys

PACKAGES = [
    "torch",
    "unsloth",
    "transformers",
    "peft",
    "trl",
    "datasets",
    "accelerate",
]


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    major, minor = sys.version_info[:2]
    if (major, minor) not in {(3, 10), (3, 11), (3, 12)}:
        print("WARNING: Unsloth is most reliable on Python 3.10-3.12.")

    for package in PACKAGES:
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, "__version__", "unknown")
            print(f"[ok] {package}: {version}")
        except Exception as exc:  # noqa: BLE001
            print(f"[missing/broken] {package}: {exc}")

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except Exception as exc:  # noqa: BLE001
        print(f"Torch CUDA check failed: {exc}")


if __name__ == "__main__":
    main()
