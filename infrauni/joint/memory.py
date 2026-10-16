"""A short streaming bank of past detection and forecast queries.

The bank is a motion prior for the current frame. It does not replace the
detection center. A zero-initialized output layer leaves the detector unchanged
until the prior is trained. Inference only reads the current frame and what
was stored from earlier frames.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .poses import apply_xy, lidar_to_lidar


class MotionPrior(nn.Module):
    """Residual attention from memory tokens into the current detection query."""

    def __init__(self, channels=256, hidden=128):
        super().__init__()
        self.query = nn.Linear(channels, hidden)
        self.key = nn.Linear(channels, hidden)
        self.value = nn.Linear(channels, hidden)
        self.position = nn.Linear(2, hidden)
        self.out = nn.Linear(hidden, channels)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, query, tokens, token_xy, query_xy, reliability):
        """Add a residual. ``query`` is ``[B, C, N]`` as TransFusion stores it."""

        if tokens is None or tokens.shape[0] == 0:
            return query
        flat = query.transpose(1, 2)
        token = tokens.to(device=flat.device, dtype=flat.dtype)
        token_xy = token_xy.to(device=flat.device, dtype=flat.dtype)
        query_xy = query_xy.to(device=flat.device, dtype=flat.dtype)
        reliability = reliability.to(device=flat.device, dtype=flat.dtype).view(1, 1, -1)
        key = self.key(token) + self.position(token_xy)
        query_embed = self.query(flat) + self.position(query_xy)
        score = torch.matmul(query_embed, key.transpose(0, 1)) / (key.shape[-1] ** 0.5)
        weight = torch.softmax(score, dim=-1) * reliability
        residual = self.out(torch.matmul(weight, self.value(token)))
        return (flat + residual).transpose(1, 2)


class QueryBank:
    """Top-scoring queries from the last few frames of one scene."""

    def __init__(self, capacity=32, history=3, step_s=0.5):
        self.capacity = capacity
        self.history = history
        self.step_s = step_s
        self.scene = None
        self.frames = []

    def reset(self):
        self.scene = None
        self.frames = []

    def observe(self, scene, timestamp, lidar_to_global, features, boxes, scores, labels, trajectories):
        """Store one frame. Tensors are detached copies."""

        if scene != self.scene:
            self.reset()
            self.scene = scene
        keep = int(min(self.capacity, scores.numel()))
        if keep == 0:
            return
        chosen = torch.topk(scores, keep).indices
        self.frames.append(
            {
                "time": float(timestamp),
                "pose": lidar_to_global,
                "features": features[chosen].detach().cpu(),
                "xy": boxes[chosen, :2].detach().cpu(),
                "scores": scores[chosen].detach().cpu(),
                "labels": labels[chosen].detach().cpu(),
                "trajectories": trajectories[chosen].detach().cpu(),
            }
        )
        self.frames = self.frames[-self.history :]

    def _observation_weight(self, frame, timestamp, lidar_to_global):
        """How well this forecast matched a later observation that is already stored."""

        later = [item for item in self.frames if frame["time"] < item["time"] < timestamp - 1e-3]
        count = frame["scores"].shape[0]
        if not later:
            return np.ones(count, dtype=np.float32)
        observed = later[-1]
        age = max(observed["time"] - frame["time"], 0.0)
        step = int(round(age / self.step_s))
        step = min(max(step, 0), frame["trajectories"].shape[2] - 1)
        predicted = apply_xy(
            lidar_to_lidar(frame["pose"], lidar_to_global),
            frame["trajectories"].numpy()[:, 0, step, :],
        )
        seen = apply_xy(lidar_to_lidar(observed["pose"], lidar_to_global), observed["xy"].numpy())
        delta = predicted[:, None, :] - seen[None, :, :]
        error = np.sqrt((delta ** 2).sum(axis=-1)).min(axis=1)
        return np.exp(-error / 2.0).astype(np.float32)

    def recall(self, mode, timestamp, lidar_to_global, reliable):
        """Return tokens whose stored time is strictly before ``timestamp``."""

        if not self.frames:
            return None
        features, xy, reliability = [], [], []
        for frame in self.frames:
            if frame["time"] >= timestamp - 1e-3:
                continue
            transform = lidar_to_lidar(frame["pose"], lidar_to_global)
            age = max(timestamp - frame["time"], 0.0)
            center = apply_xy(transform, frame["xy"].numpy())
            traj = frame["trajectories"].numpy()
            step = int(round(age / self.step_s))
            step = min(max(step, 0), traj.shape[2] - 1)
            forecast_xy = apply_xy(transform, traj[:, 0, step, :])
            spread = traj[:, :, step, :].std(axis=1).mean(axis=-1)
            consistency = 1.0 / (1.0 + spread)
            observation = self._observation_weight(frame, timestamp, lidar_to_global)
            weight = frame["scores"].numpy() * consistency * math.exp(-age / 2.0) * observation
            if not reliable:
                weight = torch.ones(len(weight)).numpy()
            feat = frame["features"]
            if mode in ("det", "both", "reliable"):
                features.append(feat)
                xy.append(torch.as_tensor(center, dtype=torch.float32))
                reliability.append(torch.as_tensor(weight, dtype=torch.float32))
            if mode in ("forecast", "both", "reliable"):
                features.append(feat)
                xy.append(torch.as_tensor(forecast_xy, dtype=torch.float32))
                reliability.append(torch.as_tensor(weight, dtype=torch.float32))
        if not features:
            return None
        return {
            "features": torch.cat(features, dim=0),
            "xy": torch.cat(xy, dim=0),
            "reliability": torch.cat(reliability, dim=0).clamp(0.0, 1.0),
        }
