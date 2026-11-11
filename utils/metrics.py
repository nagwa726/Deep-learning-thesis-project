"""
metrics.py
----------
The four metrics the report asks us to adopt (section 8):

    1. Chamfer Distance (CD)
    2. Precision
    3. Recall
    4. F1-Score

All four are computed from the SAME pair of nearest-neighbour distance
arrays, so they are always mutually consistent and use the same threshold,
exactly as section 8 requires ("يجب تحديد Threshold مناسب واستخدام القيمة
نفسها في جميع التجارب" -- a suitable threshold must be fixed and the same
value used in every experiment).

Definitions used here (matching the report's own wording in section 8):

  Let `gt`  = ground-truth complete point cloud   (Ng points)
  Let `out` = model's completed point ccloud      (No points)

  dist_gt_to_out[i]  = squared distance from gt point i    to its NEAREST point in `out`
  dist_out_to_gt[j]  = squared distance from out point j   to its NEAREST point in `gt`

  Chamfer Distance:
      cd_t (squared units, e.g. mm^2) = mean(dist_gt_to_out) + mean(dist_out_to_gt)
      cd_p (linear units,  e.g. mm)   = ( mean(sqrt(dist_gt_to_out)) + mean(sqrt(dist_out_to_gt)) ) / 2
      (same two variants as the original PointAttN repo's utils/model_utils.py::calc_cd,
       so our baseline numbers are directly comparable to the published ones)

  Recall    = fraction of GT points that were "recovered", i.e. that have
              some output point within `threshold` of them
            = mean(dist_gt_to_out < threshold**2)
            (report section 8.3: "يقيس نسبة نقاط Ground Truth التي استطاع
             النموذج إعادة بنائها ضمن threshold المحدد")

  Precision = fraction of output points that are "plausible", i.e. that
              land within `threshold` of some real GT point
            = mean(dist_out_to_gt < threshold**2)
            (report section 8.2: "يقيس نسبة النقاط الناتجة عن النموذج التي
             تقع بالقرب من نقاط Ground Truth ضمن مسافة سماح محددة")

  F1-Score  = 2 * Precision * Recall / (Precision + Recall)

NOTE ON UNITS (report section 7): `threshold` must be supplied in the SAME
units as the coordinates you pass in. If you denormalize back to physical
units (mm) before calling these functions, pass threshold in mm. If you
compute metrics in normalized space, the threshold is in normalized units
and is NOT physically meaningful on its own -- always denormalize before
reporting final thesis numbers, per the report's explicit instruction.

Everything here is pure PyTorch (no compiled CUDA extension), ported from
utils/ChamferDistancePytorch/chamfer_python.py (vendored, CPU-only) and
utils/ChamferDistancePytorch/fscore.py in the original PointAttN repo, with
our own re-derivation of which distance array corresponds to Precision vs
Recall (the original fscore.py names its two return values "precision_1"/
"precision_2" without saying which direction is which -- we made that
explicit here, since the report explicitly asks us to pin down the exact
definition of every metric before trusting it).
"""

from __future__ import annotations
import torch


def pairwise_nn_distances(gt: torch.Tensor, out: torch.Tensor):
    """
    Batched nearest-neighbour SQUARED distances in both directions.
    Pure PyTorch (CPU-friendly) -- adapted from the original repo's
    utils/ChamferDistancePytorch/chamfer_python.py::distChamfer, which is
    itself already CUDA-free.

    Args:
        gt:  (B, Ng, 3)
        out: (B, No, 3)

    Returns:
        dist_gt_to_out: (B, Ng)  squared dist from each gt point to nearest out point
        dist_out_to_gt: (B, No)  squared dist from each out point to nearest gt point
    """
    x, y = gt.double(), out.double()
    B, Ng, _ = x.shape
    _, No, _ = y.shape

    xx = torch.sum(x ** 2, dim=2)              # (B, Ng)
    yy = torch.sum(y ** 2, dim=2)              # (B, No)
    zz = torch.bmm(x, y.transpose(2, 1))       # (B, Ng, No)
    rx = xx.unsqueeze(2).expand(B, Ng, No)     # ||x_i||^2 broadcast over out-axis
    ry = yy.unsqueeze(1).expand(B, Ng, No)     # ||y_j||^2 broadcast over gt-axis
    pairwise_sq = rx + ry - 2 * zz              # (B, Ng, No); ||x_i - y_j||^2

    dist_gt_to_out = torch.min(pairwise_sq, dim=2)[0].float()  # min over out-axis
    dist_out_to_gt = torch.min(pairwise_sq, dim=1)[0].float()  # min over gt-axis
    return dist_gt_to_out, dist_out_to_gt


def chamfer_distance(gt: torch.Tensor, out: torch.Tensor):
    """
    Returns (cd_p, cd_t), both shape (B,), matching the original repo's
    calc_cd() so baseline numbers stay comparable to published PointAttN
    results:
        cd_t = mean squared distance both ways (squared units)
        cd_p = mean of sqrt(distance) both ways, averaged (linear units)
    """
    dist_gt_to_out, dist_out_to_gt = pairwise_nn_distances(gt, out)
    cd_t = dist_gt_to_out.mean(dim=1) + dist_out_to_gt.mean(dim=1)
    cd_p = (torch.sqrt(dist_gt_to_out).mean(dim=1) + torch.sqrt(dist_out_to_gt).mean(dim=1)) / 2
    return cd_p, cd_t


def precision_recall_f1(gt: torch.Tensor, out: torch.Tensor, threshold: float):
    """
    Args:
        gt, out:   (B, N, 3) point clouds, SAME units as `threshold`.
        threshold: linear-distance threshold (e.g. mm), NOT squared.

    Returns dict of (B,) tensors: precision, recall, f1
    """
    dist_gt_to_out, dist_out_to_gt = pairwise_nn_distances(gt, out)
    threshold_sq = threshold ** 2

    recall = (dist_gt_to_out < threshold_sq).float().mean(dim=1)     # GT points recovered
    precision = (dist_out_to_gt < threshold_sq).float().mean(dim=1)  # output points that are plausible

    f1 = 2 * precision * recall / (precision + recall)
    f1[torch.isnan(f1)] = 0.0  # both 0 -> define F1 as 0, not NaN

    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_batch(gt: torch.Tensor, out: torch.Tensor, threshold: float) -> dict:
    """Convenience wrapper returning every metric the report asks for, as a
    single dict of (B,) tensors: cd_p, cd_t, precision, recall, f1."""
    cd_p, cd_t = chamfer_distance(gt, out)
    pr = precision_recall_f1(gt, out, threshold)
    return {"cd_p": cd_p, "cd_t": cd_t, **pr}


if __name__ == "__main__":
    # Small, hand-checkable sanity case:
    # gt = two points far apart on the x-axis: (0,0,0) and (10,0,0)
    # out = one point that exactly matches gt point 0, and one point that is
    #       1 unit away from gt point 1.
    gt = torch.tensor([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    out = torch.tensor([[[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]]])

    cd_p, cd_t = chamfer_distance(gt, out)
    # expected: gt->out dists = [0, 1] (point (10,0,0) nearest out point is (9,0,0), dist=1)
    #           out->gt dists = [0, 1] (point (9,0,0) nearest gt point is (10,0,0), dist=1)
    # cd_t = mean([0,1]) + mean([0,1]) = 0.5 + 0.5 = 1.0
    # cd_p = (mean(sqrt([0,1])) + mean(sqrt([0,1])))/2 = (0.5 + 0.5)/2 = 0.5
    assert abs(cd_t.item() - 1.0) < 1e-6, cd_t.item()
    assert abs(cd_p.item() - 0.5) < 1e-6, cd_p.item()
    print(f"cd_p={cd_p.item():.4f} (expected 0.5), cd_t={cd_t.item():.4f} (expected 1.0)")

    # With threshold=0.5: the (10,0,0)<->(9,0,0) pair has true distance 1.0 > 0.5,
    # so it should NOT count as recovered/plausible. Only the exact-match pair counts.
    pr = precision_recall_f1(gt, out, threshold=0.5)
    assert abs(pr["recall"].item() - 0.5) < 1e-6
    assert abs(pr["precision"].item() - 0.5) < 1e-6
    assert abs(pr["f1"].item() - 0.5) < 1e-6
    print(f"threshold=0.5 -> precision={pr['precision'].item():.3f}, "
          f"recall={pr['recall'].item():.3f}, f1={pr['f1'].item():.3f} (all expected 0.5)")

    # With threshold=2.0: now the 1.0-unit gap is within threshold, so both pairs count.
    pr2 = precision_recall_f1(gt, out, threshold=2.0)
    assert abs(pr2["recall"].item() - 1.0) < 1e-6
    assert abs(pr2["precision"].item() - 1.0) < 1e-6
    print(f"threshold=2.0 -> precision={pr2['precision'].item():.3f}, "
          f"recall={pr2['recall'].item():.3f}, f1={pr2['f1'].item():.3f} (all expected 1.0)")

    print("metrics.py self-test PASSED")
