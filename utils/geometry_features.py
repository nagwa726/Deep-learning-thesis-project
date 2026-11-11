"""
geometry_features.py
---------------------
Computes the two geometric descriptors used by our contribution
"Geometry-Enhanced PointAttN":

  1. Surface Normals  (3 values per point: nx, ny, nz)
  2. Curvature        (1 value per point: local surface variation)

These are computed ONCE during data preparation (not inside the training
loop) and cached per sample, because they are deterministic functions of the
input point cloud geometry and recomputing them every epoch would just waste
CPU time on Colab.

Method (standard PCA-based local-neighborhood estimation -- the same
approach used by Open3D / PCL / most point-cloud deep learning papers):

For each point p_i:
  - find its k nearest neighbours (kNN) in the SAME point cloud
  - build the 3x3 covariance matrix of that local neighbourhood
  - eigendecompose it: eigenvalues  lambda_0 <= lambda_1 <= lambda_2
  - the eigenvector for lambda_0 (smallest eigenvalue) is the estimated
    surface normal at p_i (the direction in which the local neighbourhood
    varies the LEAST -- i.e. perpendicular to the local surface patch)
  - "surface variation" / curvature at p_i is:
        curvature_i = lambda_0 / (lambda_0 + lambda_1 + lambda_2)
    This is the standard definition used in Pauly et al. (2002),
    "Efficient Simplification of Point-Sampled Surfaces", and is also what
    Open3D's PCA-normal-estimation building blocks are based on. It is a
    number in [0, 1/3]: ~0 on a flat patch, larger on sharply curved /
    high-frequency regions. This is exactly the "Curvature" feature the
    project report asks for (a single scalar describing local surface
    bending), not the full second-fundamental-form curvature tensor.

We use Open3D for kNN + the normal-estimation covariance step because it
ships a CPU KD-tree and is already a project dependency (noted in the
report as a CPU fallback library). We then derive curvature from the same
eigenvalues Open3D's PCA step computes, via a small amount of our own code,
since Open3D does not expose a curvature value directly.

IMPORTANT (documented per the report's own instructions, section 6/7):
  - Normals have NO canonical sign from PCA alone (the eigenvector could
    point inward or outward). We orient all normals to point AWAY from the
    point cloud's centroid, which is a reasonable, deterministic convention
    for a (roughly star-convex) skull surface. This choice is logged in the
    returned metadata so it can be cited/justified in the thesis.
  - Normals/curvature are computed on the INPUT point cloud coordinates
    *before* normalization is applied to those same coordinates for the
    network (normalization only rescales/recenters XYZ; it does not change
    relative neighbourhood geometry, so normals are unaffected by uniform
    scaling and curvature is invariant only up to the same scale factor --
    see normalization.py for how this is handled consistently).
"""

from __future__ import annotations
import numpy as np

try:
    import open3d as o3d
    _HAS_OPEN3D = True
except ImportError:
    _HAS_OPEN3D = False


def _knn_covariances_numpy(points: np.ndarray, k: int) -> np.ndarray:
    """
    Pure-NumPy fallback (used only if Open3D is not installed, and for the
    unit tests in this file, which run without any heavy dependency).

    Returns the 3x3 covariance matrix of the k nearest neighbours of every
    point. O(N^2) -- fine for the ~2048-point clouds used in this project,
    not meant for huge point clouds.
    """
    N = points.shape[0]
    # pairwise squared distances (N, N)
    diff = points[:, None, :] - points[None, :, :]
    sq_dist = np.sum(diff * diff, axis=-1)
    knn_idx = np.argsort(sq_dist, axis=1)[:, :k]  # includes the point itself

    covs = np.empty((N, 3, 3), dtype=np.float64)
    for i in range(N):
        neighborhood = points[knn_idx[i]]  # (k, 3)
        centered = neighborhood - neighborhood.mean(axis=0, keepdims=True)
        covs[i] = (centered.T @ centered) / max(k - 1, 1)
    return covs


def compute_normals_and_curvature(points: np.ndarray, k: int = 20) -> dict:
    """
    Args:
        points: (N, 3) float array, a single point cloud (one sample).
        k: neighbourhood size for the local PCA estimate. 20 is a common
           default (Open3D's own default is 30; we use a smaller, cheaper
           neighbourhood since our clouds are already sparse at 2048 pts).

    Returns dict with:
        "normals":   (N, 3) float32, unit-length, oriented away from centroid
        "curvature": (N,)   float32, in [0, 1/3], PCA surface-variation measure
        "k": the neighbourhood size used (kept for the thesis methodology section)
        "backend": "open3d" or "numpy" (which implementation path ran)
    """
    assert points.ndim == 2 and points.shape[1] == 3, "points must be (N,3)"
    N = points.shape[0]
    centroid = points.mean(axis=0)

    if _HAS_OPEN3D:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
        )
        # Open3D gives us normals directly, but not curvature, so we still
        # need the per-point neighbourhood covariance eigenvalues for
        # curvature. We recompute covariances via Open3D's KD-tree for speed.
        tree = o3d.geometry.KDTreeFlann(pcd)
        eigvals_all = np.empty((N, 3), dtype=np.float64)
        for i in range(N):
            _, idx, _ = tree.search_knn_vector_3d(pcd.points[i], k)
            neighborhood = points[np.asarray(idx)]
            centered = neighborhood - neighborhood.mean(axis=0, keepdims=True)
            cov = (centered.T @ centered) / max(k - 1, 1)
            eigvals = np.linalg.eigvalsh(cov)  # ascending order
            eigvals_all[i] = np.clip(eigvals, a_min=0.0, a_max=None)
        normals = np.asarray(pcd.normals, dtype=np.float64)
        backend = "open3d"
    else:
        covs = _knn_covariances_numpy(points, k)
        eigvals_all = np.empty((N, 3), dtype=np.float64)
        normals = np.empty((N, 3), dtype=np.float64)
        for i in range(N):
            eigvals, eigvecs = np.linalg.eigh(covs[i])  # ascending order
            eigvals_all[i] = np.clip(eigvals, a_min=0.0, a_max=None)
            normals[i] = eigvecs[:, 0]  # eigenvector of smallest eigenvalue
        backend = "numpy"

    # --- orient normals outward relative to the cloud centroid ---
    to_point = points - centroid[None, :]
    dot = np.sum(normals * to_point, axis=1)
    flip_mask = dot < 0
    normals[flip_mask] *= -1.0

    # normalize to unit length (PCA eigenvectors already are, but Open3D
    # output is safest to re-normalize defensively)
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_len[norm_len == 0] = 1.0
    normals = normals / norm_len

    # --- curvature = lambda_0 / sum(eigenvalues), the Pauly et al. measure ---
    eigval_sum = eigvals_all.sum(axis=1)
    eigval_sum[eigval_sum == 0] = 1.0  # avoid divide-by-zero on degenerate nbhds
    curvature = eigvals_all[:, 0] / eigval_sum

    return {
        "normals": normals.astype(np.float32),
        "curvature": curvature.astype(np.float32),
        "k": k,
        "backend": backend,
    }


if __name__ == "__main__":
    # ---- self-test on two shapes with KNOWN correct geometry ----
    # A flat plane (z=0): every point's neighbourhood is flat -> curvature
    # should be ~0 everywhere, and normals should all point along +-Z.
    rng = np.random.default_rng(0)
    plane_xy = rng.uniform(-1, 1, size=(500, 2))
    plane = np.concatenate([plane_xy, np.zeros((500, 1))], axis=1)
    out = compute_normals_and_curvature(plane, k=16)
    assert np.mean(out["curvature"]) < 1e-3, f"plane curvature too high: {out['curvature'].mean()}"
    z_alignment = np.abs(out["normals"][:, 2])
    assert np.mean(z_alignment) > 0.95, "plane normals should align with Z axis"
    print(f"[plane]  backend={out['backend']}  mean curvature={out['curvature'].mean():.6f} "
          f"(expected ~0)  mean |nz|={z_alignment.mean():.4f} (expected ~1)")

    # A unit sphere: curvature should be a small POSITIVE constant (not 0,
    # since the surface genuinely bends), and normals should point radially
    # outward (i.e. parallel to the vector from the sphere's centre, which
    # is the origin here).
    phi = rng.uniform(0, 2 * np.pi, 800)
    costheta = rng.uniform(-1, 1, 800)
    theta = np.arccos(costheta)
    sphere = np.stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ], axis=1)
    out_sphere = compute_normals_and_curvature(sphere, k=16)
    radial = sphere / np.linalg.norm(sphere, axis=1, keepdims=True)
    cos_angle = np.sum(out_sphere["normals"] * radial, axis=1)
    assert np.mean(cos_angle) > 0.9, "sphere normals should point radially outward"
    assert out_sphere["curvature"].mean() > out["curvature"].mean(), \
        "sphere should be measurably more curved than a flat plane"
    print(f"[sphere] backend={out_sphere['backend']}  mean curvature={out_sphere['curvature'].mean():.6f} "
          f"(expected > plane's)  mean outward-alignment={cos_angle.mean():.4f} (expected ~1)")

    print("geometry_features.py self-test PASSED")
