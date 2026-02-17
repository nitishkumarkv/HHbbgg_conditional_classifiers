import torch
from contextlib import nullcontext
from typing import Optional


def get_torch_device(preferred_cuda: Optional[str] = None) -> torch.device:
    """Return the best available device, preferring CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        if preferred_cuda is not None:
            return torch.device(f'cuda:{preferred_cuda}')
        return torch.device('cuda')
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def autocast_for_device(device: torch.device):
    """Autocast context tuned to the active device; no-op for CPU."""
    if device.type == 'cuda':
        return torch.cuda.amp.autocast()
    if device.type == 'mps':
        # MPS autocast runs in float16 for better throughput on Apple Silicon.
        return torch.autocast(device_type='mps', dtype=torch.float16)
    return nullcontext()
