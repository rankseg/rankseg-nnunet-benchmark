from __future__ import annotations

import importlib.metadata
import platform
from datetime import datetime, timezone

import torch


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_provenance() -> dict:
    cuda = torch.cuda.is_available()
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "torch": torch.__version__,
            "nnunetv2": package_version("nnunetv2"),
            "rankseg": package_version("rankseg"),
            "numpy": package_version("numpy"),
            "SimpleITK": package_version("SimpleITK"),
        },
        "cuda": {
            "available": cuda,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if cuda else None,
            "device": torch.cuda.get_device_name(0) if cuda else None,
        },
    }
