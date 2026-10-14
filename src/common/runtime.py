"""Record the command, device, dependency versions and git revision of a run."""

import os
import platform
import shlex
import subprocess
import sys

import numpy as np
import torch

from .config import PROJECT_ROOT


def environment_record(device):
    """Command, dependency versions, device and git revision of the current run."""
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "command": shlex.join([sys.executable, *getattr(sys, "orig_argv", sys.argv)[1:]]),
        "python_executable": sys.executable, "python": platform.python_version(),
        "numpy": np.__version__, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "cudnn": torch.backends.cudnn.version(), "platform": platform.platform(), "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "git_revision": revision,
    }
