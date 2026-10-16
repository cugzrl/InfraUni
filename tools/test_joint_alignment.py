"""Query identity and class-aware matching. No dataset or GPU required."""

import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party", "MC-DeTra"))

import torch.nn as nn

from infrauni.eval.protocol import accumulate_forecast, self_check, summarize_forecast
from infrauni.joint.loss import apply_backward
from infrauni.joint.model import JointTransFusionDeTra
from infrauni.joint.postprocess import filter_query_indices, take_queries
from infrauni.joint.targets import attach_future, match_queries


def test_filter_keeps_identity_under_threshold_and_duplicates():
    scores = torch.tensor([0.2, 0.95, 0.05, 0.4])
    boxes = torch.tensor([
        [0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0],
        [5.0, 5.0, 0.0],
        [10.0, 0.0, 0.0],
    ])
    trajectories = torch.arange(4 * 2 * 3, dtype=torch.float32).view(4, 2, 3)
    kept = filter_query_indices(scores, boxes, score_thresh=0.1, nms_radius=1.0)
    # Query 2 is below the score gate. Query 0 is suppressed by the higher
    # score at nearly the same center. Query 3 stays.
    assert kept.tolist() == [1, 3], kept.tolist()
    gathered = take_queries(kept, trajectories=trajectories, scores=scores)
    assert torch.equal(gathered["trajectories"], trajectories[kept])
    assert torch.equal(gathered["scores"], scores[kept])


def test_range_filter_does_not_renumber_from_centers():
    scores = torch.ones(3)
    boxes = torch.tensor([
        [0.0, 0.0, 0.0],
        [80.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    kept = filter_query_indices(scores, boxes, center_range=[-5, -5, -2, 5, 5, 2])
    assert kept.tolist() == [0, 2]


def test_match_respects_class_and_distance():
    pred_xy = torch.tensor([[0.0, 0.0], [0.2, 0.0]])
    pred_labels = torch.tensor([0, 1])
    gt_xy = torch.tensor([[0.1, 0.0], [0.0, 0.1], [30.0, 0.0]])
    gt_labels = torch.tensor([1, 0, 0])
    matched = match_queries(pred_xy, pred_labels, gt_xy, gt_labels, dist_thresh=2.0)
    # Query 0 is a car and must take the car at (0, 0.1), not the nearer truck.
    assert matched.tolist() == [1, 0], matched.tolist()
    far = match_queries(
        torch.tensor([[0.0, 0.0]]),
        torch.tensor([0]),
        torch.tensor([[9.0, 0.0]]),
        torch.tensor([0]),
        dist_thresh=2.0,
    )
    assert far.tolist() == [-1]


def test_future_follows_the_matched_ground_truth():
    matched = torch.tensor([1, -1])
    gt_xy = torch.tensor([[0.0, 0.0], [4.0, 1.0]])
    gt_labels = torch.tensor([0, 1])
    track = {
        "current": torch.tensor([[0.0, 0.0], [4.0, 1.0]]).numpy(),
        "labels": torch.tensor([0, 1]).numpy(),
        "future": torch.tensor([[[1.0, 0.0]], [[8.0, 8.0]]]).numpy(),
        "mask": torch.tensor([[1.0], [1.0]]).numpy(),
    }
    future, mask = attach_future(matched, gt_xy, gt_labels, track)
    assert torch.allclose(future[0, 0], torch.tensor([8.0, 8.0]))
    assert not bool(mask[1].any())


def test_ade_ignores_invalid_future_steps():
    pred = {"labels": np.array([0]), "trajectories": np.zeros((1, 1, 4, 2))}
    pred["trajectories"][0, 0, 1] = (0, 0)
    pred["trajectories"][0, 0, 2] = (100, 0)
    pred["trajectories"][0, 0, 3] = (2, 0)
    gt = {
        "labels": np.array([0]),
        "current": np.zeros((1, 2)),
        "future": np.zeros((1, 3, 2)),
        "future_mask": np.array([[1, 0, 1]]),
    }
    car = accumulate_forecast([pred], [gt])["car"]
    # The invalid middle step is 100 m away and must not enter ADE. FDE is 2 m.
    assert abs(car["ade"] - 1.0) < 1e-6, car
    assert abs(car["fde"] - 2.0) < 1e-6, car
    summary = summarize_forecast(accumulate_forecast([pred], [gt]))
    assert summary["fp"] == 0 and summary["matched"] == 1


def test_forecast_gradient_does_not_enter_the_box_when_feedback_is_off():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.box = nn.Linear(2, 2, bias=False)
            self.future_head = nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.box.weight.copy_(torch.eye(2))
                self.future_head.weight.copy_(torch.eye(2))

        def forward(self, x):
            det = self.box(x)
            # The forecast reads the detection, so its loss would otherwise update the box.
            return det, self.future_head(det)

    torch.manual_seed(0)
    model = Tiny()
    source = torch.tensor([[1.0, 2.0]])
    det, fore = model(source)
    det_only = Tiny()
    det_only.load_state_dict(model.state_dict())
    det_ref, _ = det_only(source)
    det_ref.sum().backward()
    apply_backward(model, {"loss_det": det.sum(), "loss_forecast": fore.sum()}, feedback=False)
    assert torch.allclose(model.box.weight.grad, det_only.box.weight.grad)
    assert model.future_head.weight.grad is not None
    assert torch.allclose(model.future_head.weight.grad, torch.tensor([[1.0, 2.0], [1.0, 2.0]]))


def test_forward_does_not_accept_future_labels():
    names = JointTransFusionDeTra.forward.__code__.co_varnames
    assert "track" not in names and "gt_bboxes_3d" not in names


if __name__ == "__main__":
    test_filter_keeps_identity_under_threshold_and_duplicates()
    test_range_filter_does_not_renumber_from_centers()
    test_match_respects_class_and_distance()
    test_future_follows_the_matched_ground_truth()
    self_check()
    test_ade_ignores_invalid_future_steps()
    test_forecast_gradient_does_not_enter_the_box_when_feedback_is_off()
    test_forward_does_not_accept_future_labels()
    print("alignment checks passed")
