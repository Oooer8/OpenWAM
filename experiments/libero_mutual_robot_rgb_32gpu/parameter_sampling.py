"""Exact integer indices for validation-only parameter/gradient sampling."""
import torch


def sample_indices(numel, limit=512, *, device=None):
    if numel <= 0 or limit <= 0:
        raise ValueError('Parameter size and sampling limit must be positive')
    count = min(limit, numel)
    # Float32 linspace can round numel-1 up to numel for large FFN weights.
    # Python integer arithmetic keeps both endpoints and every index exact.
    values = [0] if count == 1 else [
        i * (numel - 1) // (count - 1) for i in range(count)
    ]
    return torch.tensor(values, dtype=torch.int64, device=device)


def sample_parameter(tensor):
    flat = tensor.detach().reshape(-1)
    indices = sample_indices(flat.numel(), device=flat.device)
    return flat.index_select(0, indices).float().cpu()
