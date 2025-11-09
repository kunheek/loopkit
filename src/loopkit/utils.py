import os
import random
from typing import List, Optional, Union

import numpy as np


def str2bool(v: Union[str, bool]) -> bool:
    """Convert string to boolean.

    Args:
        v: String or boolean value to convert

    Returns:
        Boolean value

    Raises:
        ValueError: If value cannot be converted to boolean

    Examples:
        >>> str2bool("true")
        True
        >>> str2bool("false")
        False
        >>> str2bool("1")
        True
    """
    if isinstance(v, bool):
        return v
    if not isinstance(v, str):
        raise ValueError(f"Boolean value expected, got type: {type(v).__name__}")

    v = v.strip().lower()
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise ValueError(f"Boolean value expected, got: {v}")


def _normalize_device_id(token: str) -> Optional[Union[int, str]]:
    """Normalize a device identifier from env/user input."""

    token = token.strip()
    if not token:
        return None

    try:
        return int(token)
    except ValueError:
        return token


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch if available
    try:
        import torch  # type: ignore[import-not-found]

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_gpu_devices(requested: Optional[str] = None) -> List[Union[int, str]]:
    """Get GPU device identifiers, handling numeric IDs and MIG/UUID strings."""

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        available = []
        for raw_id in cuda_visible.split(","):
            normalized = _normalize_device_id(raw_id)
            if normalized is not None:
                available.append(normalized)
    else:
        try:
            import torch  # type: ignore[import-not-found]

            available = list(range(torch.cuda.device_count()))
        except ImportError:
            available = []

    if requested:
        # Parse requested, e.g., "0,1" or "all"
        if requested.lower() == "all":
            return available
        req_ids = []
        for raw_id in requested.split(","):
            normalized = _normalize_device_id(raw_id)
            if normalized is not None:
                req_ids.append(normalized)
        return [d for d in req_ids if d in available]
    return available
