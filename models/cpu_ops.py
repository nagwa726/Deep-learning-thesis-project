"""
cpu_ops.py
----------
WHY THIS FILE EXISTS (read this before anything else):

The original PointAttN repository (https://github.com/ohhhyeahhh/PointAttN)
gets its point-sampling operations from `utils/mm3d_pn2`, which is a compiled
CUDA extension (built with `python setup.py build_ext --inplace`). Those
operations only run on a CUDA GPU. Since this project trains on Google Colab
CPU (PyTorch, no GPU), the ORIGINAL `furthest_point_sample` / `gather_points`
CANNOT be used as-is -- they will fail to import/build without a CUDA
toolchain, and even if built, cannot execute on a CPU tensor.

This file re-implements the two operations actually used by the encoder
(`furthest_point_sample`, `gather_points`) in plain PyTorch so the exact same
model architecture can run on CPU. They are numerically equivalent in spirit
to the CUDA ops (same algorithm: iterative farthest-point sampling, and
index-based gather), just slower. For a 2048-point skull with batch size
4-8, this is slow but entirely usable on CPU.

Nothing about the *model* changes because of this file -- only the backend
used to (a) pick a well-spread subset of points and (b) index tensors by
point index.
"""

import torch


def furthest_point_sample_cpu(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Iterative farthest point sampling (FPS), pure PyTorch / CPU-friendly.

    Picks `npoint` indices out of N points such that each newly picked point
    is the one farthest (in Euclidean distance) from the set already picked.
    This mirrors the semantics of mm3d_pn2's CUDA `furthest_point_sample`.

    Args:
        xyz:    (B, N, 3) float tensor of point coordinates.
        npoint: number of points to sample.

    Returns:
        (B, npoint) long tensor of sampled point indices (into the N axis).
    """
    device = xyz.device
    B, N, _ = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    # distance-to-nearest-already-picked-centroid, for every point
    distance = torch.full((B, N), 1e10, device=device)
    # start from a random point per batch item (matches typical FPS init)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)  # (B, N)
        # keep the running minimum distance to the picked set so far
        closer_mask = dist < distance
        distance[closer_mask] = dist[closer_mask]
        # next point to pick = the one currently farthest from the picked set
        farthest = torch.max(distance, dim=-1)[1]

    return centroids


def gather_points_cpu(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather channel-wise features at given point indices.
    Mirrors mm3d_pn2's CUDA `gather_points`.

    Args:
        points: (B, C, N) tensor (C channels per point, N points).
        idx:    (B, S) long tensor of indices into the N axis
                (typically the output of furthest_point_sample_cpu).

    Returns:
        (B, C, S) tensor: points[:, :, idx] done per-batch-item.
    """
    B, C, N = points.shape
    S = idx.shape[1]

    points_t = points.transpose(1, 2)  # (B, N, C)
    batch_idx = torch.arange(B, device=points.device).view(B, 1).expand(B, S)
    gathered = points_t[batch_idx, idx, :]  # (B, S, C) advanced indexing
    return gathered.transpose(1, 2).contiguous()  # (B, C, S)


# Drop-in aliases so model code can import these under the same names the
# original PointAttN code imports from utils.mm3d_pn2. This keeps the model
# file's forward() logic visually identical to the original, which makes it
# easy for a reviewer to diff this against the upstream repo.
furthest_point_sample = furthest_point_sample_cpu
gather_points = gather_points_cpu


if __name__ == "__main__":
    # tiny self-check: sampled indices must be valid and distinct-ish
    torch.manual_seed(0)
    pts = torch.randn(2, 256, 3)
    idx = furthest_point_sample_cpu(pts, 32)
    assert idx.shape == (2, 32)
    feat = torch.randn(2, 16, 256)
    g = gather_points_cpu(feat, idx)
    assert g.shape == (2, 16, 32)
    print("cpu_ops smoke check passed:", idx.shape, g.shape)
