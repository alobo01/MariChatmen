#!/usr/bin/env python
"""Validate CUDA, torch, and a small GPU matmul."""

from __future__ import annotations

import torch


def main() -> None:
    print("Torch:", torch.__version__)
    print("Torch CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to torch")

    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
    print("VRAM GB:", torch.cuda.get_device_properties(0).total_memory / 1e9)
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x
    print("CUDA matmul OK:", float(y.mean()))


if __name__ == "__main__":
    main()
