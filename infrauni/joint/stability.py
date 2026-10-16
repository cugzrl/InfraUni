"""Stop a step before a non-finite value is written into the weights."""

from __future__ import annotations

import json
import os

import torch


def first_nonfinite_param(model):
    for name, param in model.named_parameters():
        if param.requires_grad and not torch.isfinite(param.detach()).all():
            return name
    return None


def first_nonfinite_grad(model):
    for name, param in model.named_parameters():
        if param.grad is None or not param.requires_grad:
            continue
        if not torch.isfinite(param.grad).all():
            return name
    return None


def grad_norms(model, tokens):
    totals = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        for token in tokens:
            if token in name:
                totals[token] = totals.get(token, 0.0) + float(param.grad.detach().norm())
    return {key: round(value, 5) for key, value in totals.items()}


def backup_trainable(model):
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def restore_trainable(model, backup):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in backup:
                param.copy_(backup[name])


def write_failure(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
