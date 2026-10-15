"""Pick the GPU with the most free memory."""

from __future__ import annotations

import torch


def pick_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    free = []
    for index in range(torch.cuda.device_count()):
        free_bytes, _ = torch.cuda.mem_get_info(index)
        free.append(free_bytes)
    choice = max(range(len(free)), key=lambda index: free[index])
    return torch.device(f"cuda:{choice}")
