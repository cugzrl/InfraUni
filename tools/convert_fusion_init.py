"""Adapt MSMDFusion's merged init to this TransFusion spconv layout.

Official SparseConvolution stores weights as (kD, kH, kW, in, out).
The released fusion checkpoint stores the same kernels as (out, kD, kH, kW, in).
"""

import argparse
from collections import Counter

import torch


def _convert_kernel(value, target_shape):
    if tuple(value.shape) == tuple(target_shape):
        return value
    if value.ndim == 5 and tuple(value.permute(1, 2, 3, 4, 0).shape) == tuple(target_shape):
        return value.permute(1, 2, 3, 4, 0).contiguous()
    return None


def convert(src, dst, model_state):
    checkpoint = torch.load(src, map_location="cpu")
    state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    converted = {}
    changed = []
    bad = []
    for key, value in state.items():
        name = key[7:] if key.startswith("module.") else key
        if name not in model_state:
            converted[name] = value
            continue
        new_value = _convert_kernel(value, model_state[name].shape)
        if new_value is None:
            bad.append((name, tuple(value.shape), tuple(model_state[name].shape)))
            continue
        if tuple(new_value.shape) != tuple(value.shape):
            changed.append(name)
        converted[name] = new_value
    if bad:
        raise RuntimeError("unconverted kernels: %s" % bad)
    meta = dict(checkpoint.get("meta") or {})
    meta["source"] = src
    meta["spconv_layout"] = "kD_kH_kW_in_out"
    torch.save({"meta": meta, "state_dict": converted}, dst)
    return converted, changed


def report(converted, model):
    model_state = model.state_dict()
    missing = sorted(set(model_state) - set(converted))
    unexpected = sorted(set(converted) - set(model_state))
    print("matched", len(set(model_state) & set(converted)))
    print("missing", len(missing))
    print("unexpected", len(unexpected))
    print("missing groups", Counter(".".join(k.split(".")[:3]) for k in missing))
    frozen_missing = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            continue
        if name not in converted:
            frozen_missing.append(name)
    print("frozen_but_missing", len(frozen_missing))
    for name in frozen_missing[:20]:
        print(" ", name)
    return missing, unexpected, frozen_missing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    # Imported only when executed as a script, after PYTHONPATH is set.
    import logging

    logging.getLogger("mmcv").setLevel(logging.ERROR)
    from mmcv import Config
    from mmdet3d.models import build_detector

    cfg = Config.fromfile(args.config)
    model = build_detector(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    if cfg.get("freeze_lidar_components", False):
        for name, param in model.named_parameters():
            if "pts" in name and "pts_bbox_head" not in name:
                param.requires_grad = False
            if "pts_bbox_head.decoder.0" in name:
                param.requires_grad = False
            if "pts_bbox_head.shared_conv" in name and "shared_conv_img" not in name:
                param.requires_grad = False
            if "pts_bbox_head.heatmap_head" in name and "heatmap_head_img" not in name:
                param.requires_grad = False
            if "pts_bbox_head.prediction_heads.0" in name:
                param.requires_grad = False
            if "pts_bbox_head.class_encoding" in name:
                param.requires_grad = False
    converted, changed = convert(args.src, args.dst, model.state_dict())
    print("permuted_kernels", len(changed))
    report(converted, model)
