import json
import random
from pathlib import Path

import numpy as np
import torch


def load_json(path):
    """Load a UTF-8 JSON configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed):
    """Seed Python, NumPy, and PyTorch random generators for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    """Return CPU when CUDA is requested but unavailable."""
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def ensure_dir(path):
    """Create a directory tree if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def batch_iter(items, batch_size, shuffle=True):
    """Yield mini-batches from a Python sequence."""
    idx = list(range(len(items)))
    if shuffle:
        random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        yield [items[i] for i in idx[start:start + batch_size]]
 