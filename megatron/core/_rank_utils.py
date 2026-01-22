# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Low-level rank utilities with minimal dependencies to avoid circular imports."""

import os

import torch


def safe_get_rank() -> int:
    """Safely get the rank of the current process.

    Returns the rank from torch.distributed if initialized, otherwise falls back
    to the RANK environment variable, defaulting to 0.

    Returns:
        int: The rank of the current process.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()

    # If torch.distributed is not initialized, try to read environment variables.
    try:
        return int(os.environ.get("RANK", 0))
    except (ValueError, TypeError):
        return 0
