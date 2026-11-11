"""
normalization.py
-----------------
Implements exactly the Normalization / Denormalization procedure described
in the project report (section 6):

    P_normalized = (P_original - Center) / Scale
    P_original   = P_normalized * Scale + Center

- Center: centroid of the COMPLETE (ground-truth) skull's points, so the
  partial input and the completed output both get denormalized back into
  the same coordinate frame as the original full skull.
- Scale: a single scalar per sample (not per-axis), computed as the max
  distance from any point to the centroid, AFTER centering. Using one
  scalar (not 3 separate per-axis scales) is important: it keeps the shape
  undistorted (no non-uniform stretching), which the report explicitly
  cares about ("المحافظة على التفاصيل والانحناءات الهندسية" -- preserve
  geometric detail/curvature). Non-uniform axis scaling would silently
  distort curvature.

Center and Scale are computed from the COMPLETE point cloud and then reused
to normalize the corresponding INCOMPLETE (partial) point cloud of the same
sample, so both live in the exact same normalized frame (this matters
because the network takes the partial cloud as input and is supervised
against the complete cloud -- they must be normalized consistently, with
the same Center/Scale, or the task becomes ill-posed).

Center and Scale are saved per sample (see save_norm_params / load_norm_params)
so evaluation can always recover physical units later, per the report's
explicit requirement in section 7 (do not report a physical unit for any
metric before confirming the raw data's real units, e.g. mm vs cm for the
original PLY/VTK files).
"""

from __future__ import annotations
import json
import numpy as np


def compute_center_scale(complete_points: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Args:
        complete_points: (N, 3) array, the COMPLETE (ground-truth) point cloud.

    Returns:
        center: (3,) float64 array
        scale:  python float (scalar, > 0)
    """
    center = complete_points.mean(axis=0)
    centered = complete_points - center[None, :]
    radii = np.linalg.norm(centered, axis=1)
    scale = float(radii.max())
    if scale <= 0:
        # degenerate (e.g. a single repeated point) -- avoid div-by-zero
        scale = 1.0
    return center, scale


def normalize(points: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    """P_normalized = (P_original - Center) / Scale"""
    return (points - center[None, :]) / scale


def denormalize(points_normalized: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    """P_original = P_normalized * Scale + Center"""
    return points_normalized * scale + center[None, :]


def denormalize_curvature(curvature_normalized: np.ndarray, scale: float) -> np.ndarray:
    """
    Curvature (Pauly et al. lambda0/sum(lambda) ratio) is already
    SCALE-INVARIANT by construction (it's a ratio of eigenvalues of the same
    units), so denormalizing curvature is a no-op mathematically. This
    function exists only so the training/eval code has one explicit,
    documented place that states that fact, instead of silently assuming it
    -- the report asks us not to assume unit behaviour without checking it.
    """
    return curvature_normalized  # scale cancels out in the eigenvalue ratio


def save_norm_params(path: str, center: np.ndarray, scale: float, extra: dict | None = None) -> None:
    payload = {"center": center.tolist(), "scale": float(scale)}
    if extra:
        payload.update(extra)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_norm_params(path: str) -> tuple[np.ndarray, float, dict]:
    with open(path) as f:
        payload = json.load(f)
    center = np.array(payload["center"], dtype=np.float64)
    scale = float(payload["scale"])
    return center, scale, payload


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    pts = rng.uniform(-5, 5, size=(2048, 3)) + np.array([100.0, -40.0, 7.5])  # offset, like real skull coords
    center, scale = compute_center_scale(pts)

    normed = normalize(pts, center, scale)
    assert np.max(np.linalg.norm(normed, axis=1)) <= 1.0 + 1e-6, "normalized radius should be <= 1"

    recovered = denormalize(normed, center, scale)
    max_err = np.max(np.abs(recovered - pts))
    assert max_err < 1e-8, f"round-trip normalize/denormalize failed, max_err={max_err}"

    print(f"center={center}, scale={scale:.4f}")
    print(f"round-trip max abs error = {max_err:.2e} (expected ~0)")
    print("normalization.py self-test PASSED")
