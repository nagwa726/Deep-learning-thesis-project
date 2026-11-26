"""
stl_io.py
----------
Minimal STL mesh reading + surface point-sampling, in pure NumPy.

WHY THIS FILE EXISTS: the real MUG500+ dataset does not ship ready-made
point clouds. Per the dataset's own data-descriptor paper (Li et al.,
"MUG500+: Database of 500 high-resolution healthy human skulls...", Data
in Brief, 2021): each skull folder contains a `.nrrd` CT volume AND a
`.stl` surface mesh of that same skull (the `.stl` is "the corresponding
mesh of the skull" -- already segmented, no need to threshold/march-cubes
the NRRD ourselves). So `prepare_mug500.py` needs to read `.stl` files and
sample points off their surface, not read point clouds directly.

This is implemented here in plain NumPy (not via Open3D's STL+sampling
helpers) specifically so it could be unit-tested in a sandbox that has
NumPy but not Open3D/PyTorch -- see this file's __main__ block, which
builds a tiny synthetic mesh with a mathematically known surface area and
checks the sampler's output against it.

Supports BINARY STL (the common case for meshes exported by segmentation
tools like the ones used to produce MUG500+) and falls back to ASCII STL
if the binary magic/size check fails.
"""

from __future__ import annotations
import struct
import numpy as np


def read_stl(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads an STL file (binary or ASCII).

    Returns:
        vertices: (V, 3) float64 array of unique vertex positions
        faces:    (F, 3) int64 array of vertex indices per triangle
    """
    with open(path, "rb") as f:
        header = f.read(80)
        rest = f.read()

    # Binary STL: 80-byte header, 4-byte uint32 triangle count, then
    # 50 bytes/triangle (12 floats for normal+3 verts = 12*4=48, +2 byte
    # "attribute byte count"). Verify the file size matches exactly --
    # ASCII STL files that happen to start with "solid" can otherwise be
    # misread as binary.
    if len(rest) >= 4:
        (n_tri,) = struct.unpack("<I", rest[:4])
        expected_size = 80 + 4 + n_tri * 50
        with open(path, "rb") as f:
            actual_size = len(f.read())
        if actual_size == expected_size and n_tri > 0:
            return _read_stl_binary(rest[4:], n_tri)

    return _read_stl_ascii(path)


def _read_stl_binary(buf: bytes, n_tri: int) -> tuple[np.ndarray, np.ndarray]:
    tri_verts = np.empty((n_tri, 3, 3), dtype=np.float64)
    offset = 0
    rec_size = 50
    for i in range(n_tri):
        rec = buf[offset:offset + rec_size]
        # 12 bytes normal (skip), then 3 vertices x 3 floats x 4 bytes
        v = struct.unpack("<9f", rec[12:48])
        tri_verts[i, 0] = v[0:3]
        tri_verts[i, 1] = v[3:6]
        tri_verts[i, 2] = v[6:9]
        offset += rec_size
    return _dedup_vertices(tri_verts)


def _read_stl_ascii(path: str) -> tuple[np.ndarray, np.ndarray]:
    verts = []
    tri = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("vertex"):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                if len(verts) % 3 == 0:
                    tri.append(verts[-3:])
    tri_verts = np.array(tri, dtype=np.float64)
    return _dedup_vertices(tri_verts)


def _dedup_vertices(tri_verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """tri_verts: (F, 3, 3) raw per-triangle vertex coordinates (with
    duplicates, since STL repeats shared vertices). Returns deduplicated
    (vertices, faces) -- not strictly required for sampling (which only
    needs tri_verts), but a more useful mesh representation to return."""
    flat = tri_verts.reshape(-1, 3)
    unique_verts, inverse = np.unique(flat, axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    return unique_verts, faces


def sample_points_from_mesh(vertices: np.ndarray, faces: np.ndarray,
                             n_points: int, rng: np.random.Generator) -> np.ndarray:
    """
    Area-weighted random surface sampling (the standard way to get a
    roughly-uniform point cloud off a triangle mesh -- same idea as
    Open3D's sample_points_uniformly).

    Args:
        vertices: (V, 3)
        faces: (F, 3) int indices into vertices
        n_points: how many points to sample
        rng: numpy Generator (for reproducibility)

    Returns:
        (n_points, 3) float64 array of points lying exactly on the mesh surface.
    """
    tri = vertices[faces]  # (F, 3, 3)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    cross = np.cross(b - a, c - a)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total_area = areas.sum()
    if total_area <= 0:
        raise ValueError("mesh has zero total surface area -- degenerate/empty mesh")
    probs = areas / total_area

    tri_idx = rng.choice(len(faces), size=n_points, p=probs)
    # random barycentric coordinates (uniform over the triangle)
    r1 = rng.random(n_points)
    r2 = rng.random(n_points)
    sqrt_r1 = np.sqrt(r1)
    u = 1 - sqrt_r1
    v = sqrt_r1 * (1 - r2)
    w = sqrt_r1 * r2

    chosen = tri[tri_idx]  # (n_points, 3, 3)
    points = (u[:, None] * chosen[:, 0] + v[:, None] * chosen[:, 1] + w[:, None] * chosen[:, 2])
    return points


def load_mesh_and_sample(path: str, n_points: int, seed: int | None = None) -> np.ndarray:
    """Convenience: read_stl + sample_points_from_mesh in one call."""
    vertices, faces = read_stl(path)
    rng = np.random.default_rng(seed)
    return sample_points_from_mesh(vertices, faces, n_points, rng)


# ---------------------------------------------------------------------
# Tiny STL writer, used ONLY by the self-test below to generate a
# synthetic mesh with a mathematically known surface area (a unit cube),
# so sample_points_from_mesh can be checked against ground truth without
# needing any real skull data.
# ---------------------------------------------------------------------
def _write_binary_stl_cube(path: str, size: float = 1.0) -> None:
    s = size
    # 8 cube corners
    v = np.array([
        [0, 0, 0], [s, 0, 0], [s, s, 0], [0, s, 0],
        [0, 0, s], [s, 0, s], [s, s, s], [0, s, s],
    ], dtype=np.float64)
    # 12 triangles (2 per face), consistent winding not required for our sampler
    faces = [
        (0, 1, 2), (0, 2, 3),  # bottom
        (4, 6, 5), (4, 7, 6),  # top
        (0, 4, 5), (0, 5, 1),  # side
        (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3),
        (3, 7, 4), (3, 4, 0),
    ]
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", len(faces)))
        for (i, j, k) in faces:
            normal = (0.0, 0.0, 0.0)  # unused by our reader
            f.write(struct.pack("<3f", *normal))
            for idx in (i, j, k):
                f.write(struct.pack("<3f", *v[idx]))
            f.write(struct.pack("<H", 0))


if __name__ == "__main__":
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmp:
        cube_path = os.path.join(tmp, "unit_cube.stl")
        _write_binary_stl_cube(cube_path, size=2.0)  # 2x2x2 cube, surface area = 6*(2*2) = 24

        vertices, faces = read_stl(cube_path)
        assert vertices.shape == (8, 3), vertices.shape
        assert faces.shape == (12, 3), faces.shape
        print(f"read_stl: {vertices.shape[0]} vertices, {faces.shape[0]} faces (expected 8, 12)")

        rng = np.random.default_rng(0)
        n = 20000
        pts = sample_points_from_mesh(vertices, faces, n, rng)
        assert pts.shape == (n, 3)

        # Every sampled point must lie exactly on one of the cube's 6 faces
        # (i.e. at least one coordinate is ~0 or ~2, the rest within [0,2]).
        on_surface = (
            np.any(np.isclose(pts, 0.0, atol=1e-9) | np.isclose(pts, 2.0, atol=1e-9), axis=1)
        )
        in_bounds = np.all((pts >= -1e-9) & (pts <= 2.0 + 1e-9), axis=1)
        assert on_surface.all(), f"{(~on_surface).sum()} points are not on any cube face"
        assert in_bounds.all(), "some points fell outside the cube's bounding box"
        print(f"all {n} sampled points lie exactly on the cube's surface (on a face, in bounds)")

        # Area-weighted sampling on a cube (all 6 faces equal area) should
        # put ~1/6 of points on each face -- check the face at x=2 gets a
        # fraction close to 1/6 (loose tolerance since this is random).
        frac_x2 = np.isclose(pts[:, 0], 2.0, atol=1e-9).mean()
        print(f"fraction of points on the x=2 face: {frac_x2:.4f} (expected ~0.1667)")
        assert abs(frac_x2 - 1 / 6) < 0.02, f"sampling is not area-uniform: {frac_x2}"

        print("\nstl_io.py self-test PASSED (binary STL read + area-weighted surface sampling verified).")
