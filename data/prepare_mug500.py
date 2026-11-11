"""
prepare_mug500.py
-------------------
Offline data-preparation script for the MUG500+ skull dataset (section 5 of
the project report). Run this ONCE (in Colab, after downloading and
extracting the real dataset from figshare) to turn raw complete skull PLY
files into fixed-size, normalized (complete, incomplete, normals,
curvature) training samples saved as .npz, so the training scripts
themselves stay fast and simple (no PLY I/O / geometry computation in the
training loop).

WHAT THIS SCRIPT DOES, STEP BY STEP (matches report section 5 exactly):

 1. Loads each COMPLETE ("A-case", i.e. non-defective) skull PLY file.
    -> report: "في تجاربنا نستخدم الحالات السليمة (A-cases)."
 2. The raw skull has 8192 points. We farthest-point-sample it down to 2048
    points -- this 2048-point cloud is the "complete" ground truth used
    for training.
    -> report: "عدد النقاط في النموذج الأصلي: 8192 ... المستخدم في تجهيز
       بيانات التدريب: 2048."
 3. We create the INCOMPLETE version by removing a spatially-contiguous
    region from ONE side of the skull along the X axis (NOT randomly
    scattered points), using an X-axis percentile cutoff, targeting ~30%
    removed / ~70% kept.
    -> report: "نستخدم إزالة مكانية لمنطقة من جهة واحدة ... تم استخدام
       X-percentile ... الفكرة هي محاكاة فقدان جزء متصل (Region-based /
       Structured Missingness)."
 4. We VERIFY the actual resulting kept-fraction (it will not be exactly
    70% because points aren't uniformly distributed along X) and record it
    per sample, rather than assuming it.
    -> report: "يجب التحقق من عدد النقاط الفعلي ... وعدم افتراض أن النسبة
       النهائية دائماً 30% بالضبط دون فحص الكود."
 5. We compute Normalization parameters (center, scale) from the COMPLETE
    cloud, and apply the SAME parameters to the incomplete cloud (see
    utils/normalization.py), so both are consistent.
 6. We compute Surface Normals + Curvature (utils/geometry_features.py) on
    the NORMALIZED incomplete cloud (since that is what the network will
    actually receive as input).
 7. We resize the incomplete cloud to a FIXED point count (so samples can
    be batched), using the exact same tile-then-random-pad / random-subsample
    technique the upstream PointAttN repo's own dataset.py already uses
    (see `_resize_to_fixed_count` below) -- not a new, unjustified technique.
 8. Everything is saved per-sample as one .npz file plus one shared
    manifest.json (train/val/test split).

IMPORTANT, per report section 11 ("مهم جدا"): raw/processed skull point
files are MEDICAL DATA and must NOT be pushed to the public GitHub repo.
This script writes its output under `data/processed/` which is listed in
.gitignore -- only the CODE that produces that data is tracked in git.

Usage (after placing extracted PLY files under --raw_dir):
    python data/prepare_mug500.py \
        --raw_dir  /path/to/MUG500plus/complete_skulls \
        --out_dir  data/processed/mug500 \
        --n_complete 2048 --missing_ratio 0.30 --knn_geo 20

Run with --self_test to generate a handful of SYNTHETIC ellipsoid "skulls"
instead of reading real PLY files -- useful to verify the whole pipeline
runs correctly before the real dataset is available (see tests/).
"""

from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.normalization import compute_center_scale, normalize, save_norm_params
from utils.geometry_features import compute_normals_and_curvature


def farthest_point_sample_np(points: np.ndarray, n_sample: int, seed: int | None = None) -> np.ndarray:
    """Simple O(N*n_sample) NumPy FPS, used only here at data-prep time (a
    one-off cost per sample), as opposed to models/cpu_ops.py's batched
    torch version used at training time."""
    rng = np.random.default_rng(seed)
    N = points.shape[0]
    if n_sample >= N:
        return np.arange(N)
    chosen = np.empty(n_sample, dtype=np.int64)
    chosen[0] = rng.integers(0, N)
    dist = np.full(N, np.inf)
    for i in range(1, n_sample):
        last = points[chosen[i - 1]]
        d = np.sum((points - last) ** 2, axis=1)
        dist = np.minimum(dist, d)
        chosen[i] = np.argmax(dist)
    return chosen


def remove_region_x_percentile(points: np.ndarray, missing_ratio: float, side: str, rng) -> tuple[np.ndarray, float]:
    """
    Structured / region-based missingness (report section 5):
    removes the points on one side of an X-axis cutoff chosen so that
    (approximately) `missing_ratio` fraction of points fall on the removed
    side. `side` is "low" (remove the low-X end) or "high" (remove the
    high-X end) -- chosen per sample so the model sees defects from both
    directions across the dataset, matching real defect variability,
    while the ratio stays fixed as the report requires.

    Returns (kept_points, actual_kept_fraction). The actual fraction is
    reported explicitly rather than assumed, per the report's own caution.
    """
    x = points[:, 0]
    if side == "low":
        cutoff = np.percentile(x, missing_ratio * 100.0)
        keep_mask = x >= cutoff
    elif side == "high":
        cutoff = np.percentile(x, (1.0 - missing_ratio) * 100.0)
        keep_mask = x <= cutoff
    else:
        raise ValueError(side)
    kept = points[keep_mask]
    actual_kept_fraction = kept.shape[0] / points.shape[0]
    return kept, actual_kept_fraction


def resize_to_fixed_count(points: np.ndarray, n_target: int, rng) -> np.ndarray:
    """
    Resize a point cloud to EXACTLY n_target points, so samples can be
    batched as fixed-size tensors. Same technique as upstream PointAttN's
    own dataset.py (`randomsample` / `upsample` methods): random subsample
    if we have more than needed, tile + random-pad if we have fewer. This
    is not a new invented technique -- it is literally what the baseline
    repo already does for its own (PCN/Completion3D) partial clouds, reused
    here for consistency.
    """
    n = points.shape[0]
    if n == n_target:
        return points
    if n > n_target:
        choice = rng.permutation(n)[:n_target]
        return points[choice]
    # n < n_target: tile, then randomly pad the remainder
    reps = n_target // n
    remainder = n_target - reps * n
    tiled = np.tile(points, (reps, 1))
    if remainder > 0:
        pad_idx = rng.permutation(n)[:remainder]
        tiled = np.concatenate([tiled, points[pad_idx]], axis=0)
    return tiled


def process_one_skull(points_8192: np.ndarray, sample_id: str, args, rng) -> dict:
    """Runs steps 2-7 of the module docstring on a single raw skull."""
    # Step 2: 8192 -> 2048, FPS for good spatial coverage
    idx = farthest_point_sample_np(points_8192, args.n_complete, seed=rng.integers(1 << 30))
    complete = points_8192[idx].astype(np.float64)

    # Step 3/4: structured, one-sided region removal with verified ratio
    side = rng.choice(["low", "high"])
    partial_raw, actual_kept_fraction = remove_region_x_percentile(
        complete, args.missing_ratio, side, rng
    )

    # Step 5: normalization params from the COMPLETE cloud, applied to both
    center, scale = compute_center_scale(complete)
    complete_norm = normalize(complete, center, scale)
    partial_norm_raw = normalize(partial_raw, center, scale)

    # Step 7: fixed-size partial input (so batches can be stacked)
    n_partial_target = int(round((1.0 - args.missing_ratio) * args.n_complete))
    partial_norm = resize_to_fixed_count(partial_norm_raw, n_partial_target, rng)

    # Step 6: geometric descriptors computed on the (normalized) PARTIAL
    # input actually fed to the network
    geo = compute_normals_and_curvature(partial_norm.astype(np.float64), k=args.knn_geo)

    return {
        "sample_id": sample_id,
        "complete": complete_norm.astype(np.float32),           # (n_complete, 3)
        "partial": partial_norm.astype(np.float32),              # (n_partial_target, 3)
        "normals": geo["normals"],                                # (n_partial_target, 3)
        "curvature": geo["curvature"],                            # (n_partial_target,)
        "center": center, "scale": scale,
        "removal_side": side,
        "target_missing_ratio": args.missing_ratio,
        "actual_kept_fraction_before_resize": actual_kept_fraction,
        "geo_backend": geo["backend"],
    }


def _save_sample(sample: dict, out_dir: str) -> str:
    path = os.path.join(out_dir, f"{sample['sample_id']}.npz")
    np.savez_compressed(
        path,
        complete=sample["complete"], partial=sample["partial"],
        normals=sample["normals"], curvature=sample["curvature"],
    )
    save_norm_params(
        path.replace(".npz", "_norm.json"), sample["center"], sample["scale"],
        extra={
            "removal_side": sample["removal_side"],
            "target_missing_ratio": sample["target_missing_ratio"],
            "actual_kept_fraction_before_resize": sample["actual_kept_fraction_before_resize"],
            "geo_backend": sample["geo_backend"],
        },
    )
    return path


def _load_ply_points(path: str) -> np.ndarray:
    try:
        import open3d as o3d
    except ImportError as e:
        raise RuntimeError(
            "open3d is required to read real MUG500+ PLY files. "
            "Install it in your Colab environment (`pip install open3d`)."
        ) from e
    pcd = o3d.io.read_point_cloud(path)
    return np.asarray(pcd.points, dtype=np.float64)


def make_synthetic_skull(n_points: int, rng) -> np.ndarray:
    """
    Generates a bumpy-ellipsoid point cloud standing in for a real skull,
    used ONLY by --self_test (and tests/test_pipeline_smoke.py) so the
    whole data-prep + training pipeline can be exercised without the real,
    restricted MUG500+ dataset. Not used for anything reported in the
    thesis -- purely an engineering smoke test.
    """
    phi = rng.uniform(0, 2 * np.pi, n_points)
    costheta = rng.uniform(-1, 1, n_points)
    theta = np.arccos(costheta)
    base = np.stack([
        np.sin(theta) * np.cos(phi) * 90,   # rough skull-scale half-axes in mm
        np.sin(theta) * np.sin(phi) * 70,
        np.cos(theta) * 100,
    ], axis=1)
    bumps = 4.0 * np.sin(4 * phi)[:, None] * np.array([[1, 0, 0]]) * rng.normal(1, 0.1, (n_points, 1))
    return base + bumps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", type=str, default=None, help="dir of raw complete-skull .ply files")
    p.add_argument("--out_dir", type=str, default="data/processed/mug500")
    p.add_argument("--n_complete", type=int, default=2048)
    p.add_argument("--missing_ratio", type=float, default=0.30)
    p.add_argument("--knn_geo", type=int, default=20)
    p.add_argument("--val_frac", type=float, default=0.10)
    p.add_argument("--test_frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--self_test", action="store_true",
                    help="use synthetic ellipsoid skulls instead of --raw_dir (no real data needed)")
    p.add_argument("--self_test_n", type=int, default=12)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if args.self_test:
        sample_ids = [f"synthetic_{i:04d}" for i in range(args.self_test_n)]
        raw_clouds = {sid: make_synthetic_skull(8192, rng) for sid in sample_ids}
    else:
        if not args.raw_dir:
            raise SystemExit("--raw_dir is required unless --self_test is set")
        files = sorted(f for f in os.listdir(args.raw_dir) if f.lower().endswith(".ply"))
        sample_ids = [os.path.splitext(f)[0] for f in files]
        raw_clouds = {sid: _load_ply_points(os.path.join(args.raw_dir, f))
                      for sid, f in zip(sample_ids, files)}

    manifest = {"samples": [], "args": vars(args)}
    for sid in sample_ids:
        sample = process_one_skull(raw_clouds[sid], sid, args, rng)
        path = _save_sample(sample, args.out_dir)
        manifest["samples"].append({
            "id": sid, "path": os.path.basename(path),
            "actual_kept_fraction_before_resize": sample["actual_kept_fraction_before_resize"],
        })
        print(f"[prepare_mug500] {sid}: kept_fraction(before resize)="
              f"{sample['actual_kept_fraction_before_resize']:.3f} "
              f"(target={1 - args.missing_ratio:.2f})")

    # deterministic train/val/test split
    ids = [s["id"] for s in manifest["samples"]]
    rng.shuffle(ids)
    n = len(ids)
    n_val = max(1, int(round(n * args.val_frac))) if n > 2 else 0
    n_test = max(1, int(round(n * args.test_frac))) if n > 2 else 0
    split = {
        "test": ids[:n_test],
        "val": ids[n_test:n_test + n_val],
        "train": ids[n_test + n_val:],
    }
    manifest["split"] = split

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {n} samples to {args.out_dir}")
    print(f"split sizes -> train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")


if __name__ == "__main__":
    main()
