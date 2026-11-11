# Geometry-Enhanced PointAttN — 3D Anatomical Point Cloud Completion

Master's thesis project: completing missing regions of 3D human skull point
clouds with deep learning, built on top of the **PointAttN** baseline
(Wang et al., AAAI 2024), with a proposed encoder modification that fuses
**Surface Normals** and **Curvature** into the model's feature extraction.

This README explains the project end to end: the research idea, the exact
architectural change, the data pipeline, how to run everything, and what is
still outstanding before real results can be produced.

## 1. Research idea

Each skull is a 3D point cloud (X, Y, Z per point). We take a complete
skull, remove a spatially-contiguous region to simulate missing bone, and
train a deep network to reconstruct the missing part so the output matches
the original (ground truth) as closely as possible — preserving fine
geometric detail and curvature, not just the rough outline.

## 2. Baseline model

**PointAttN: You Only Need Attention for Point Cloud Completion**
Wang, Cui, Guo, Li, Liu, Shen — AAAI 2024. DOI: 10.1609/aaai.v38i6.28356.
Upstream code: https://github.com/ohhhyeahhh/PointAttN

We keep the original architecture and completion mechanism unchanged for
the baseline (Experiment 1) — see `LICENSE_NOTE.md` for exactly what was
adapted and why (CPU compatibility only, not an architectural change).

## 3. Our contribution: Geometry-Enhanced PointAttN

We add two geometric descriptors, computed per point:

- **Surface Normals** (3 values): local surface orientation.
- **Curvature** (1 value): local surface bending (Pauly et al.'s PCA
  surface-variation measure, `λ₀ / (λ₀+λ₁+λ₂)` from the point's local
  neighbourhood covariance).

**Where the change lives (the most important design decision):** both
features are concatenated with XYZ and fed into the Encoder's *first*
convolution (`conv1`), i.e. fused at the very start of feature extraction —
**not** appended late in the pipeline. Point sampling/gathering inside the
encoder still operates on plain XYZ, so the network's spatial reasoning is
unaffected; only the *per-point feature content* changes. See the detailed
rationale in `models/pointattn_geo.py`'s docstring, and compare
`PCT_encoder.forward` (baseline) with `PCT_encoder_geo.forward`
(contribution) side by side — the diff is intentionally tiny and localized.

## 4. Ablation study (4 experiments, same code/config otherwise)

| # | Model                                   | `geo_mode` | Script                   | Config                                  |
|---|------------------------------------------|-----------|--------------------------|------------------------------------------|
| 1 | Original PointAttN                        | `none`    | `train_baseline.py`      | `configs/exp1_original.yaml`             |
| 2 | PointAttN + Surface Normals               | `normals` | `train_contribution.py`  | `configs/exp2_normals.yaml`              |
| 3 | PointAttN + Curvature                     | `curvature` | `train_contribution.py` | `configs/exp3_curvature.yaml`            |
| 4 | **Geometry-Enhanced PointAttN** (ours)     | `both`    | `train_contribution.py`  | `configs/exp4_geometry_enhanced.yaml`    |

All four share identical batch size, epochs, optimizer, learning-rate
schedule, F1 threshold and missingness generation — only the encoder input
differs — so the comparison is fair.

## 5. Datasets

- **MUG500+** (primary): 500 intact skulls (we use the "A-cases"/healthy
  ones). Original resolution 8192 points/skull, subsampled to 2048 for
  training. Missing region = one-sided, X-percentile-based removal
  targeting ~30% missing / ~70% kept (NOT random point deletion — this
  simulates a spatially contiguous real-world defect).
- **PPCNet** (secondary, generalization check): lumbar vertebrae L1–L5,
  pre-split 829/103/105 (train/val/test). Kept and reported as an
  **independent** experiment — never mixed with MUG500+ numbers.

Run `data/prepare_mug500.py` once to turn raw PLY files into the normalized,
fixed-size `.npz` samples the training scripts consume (see "How to run"
below). **Medical point-cloud data is never committed to this repo** —
`data/processed/` and `data/raw/` are gitignored, per the report's own
requirement.

## 6. Normalization, denormalization, physical units

- Normalization: `P_normalized = (P_original - Center) / Scale`, with
  `Center` = centroid and `Scale` = max radius of the *complete* skull
  (one scalar, not per-axis, so shape/curvature isn't distorted).
  `Center`/`Scale` are saved per sample (`*_norm.json`).
- Before computing any metric, both the completed output and the ground
  truth are denormalized back to physical units (`utils/normalization.py`,
  `denormalize()`), using that sample's own saved Center/Scale.
- **Units are not assumed.** Before trusting "mm" anywhere in the thesis,
  confirm the real unit of the original MUG500+ PLY files / PPCNet VTK
  files from their documentation — do not assume mm vs cm. Curvature itself
  is a dimensionless eigenvalue ratio and needs no unit conversion
  (see `denormalize_curvature()`).

## 7. Metrics (`utils/metrics.py`)

Chamfer Distance (`cd_p` linear units, `cd_t` squared units), Precision,
Recall, F1-Score — all derived from the same pair of nearest-neighbour
distance arrays and the same fixed threshold across every experiment, with
each metric's exact definition spelled out in the module docstring (this
mirrors the report's explicit instruction to pin down what "precision" and
"recall" mean rather than trust the baseline repo's ambiguous variable
names).

## 8. Repository layout

```
configs/                 one YAML per ablation experiment
data/
  prepare_mug500.py       offline data-prep (raw PLY -> normalized .npz)
  skull_dataset.py        PyTorch Dataset over the prepared .npz files
models/
  cpu_ops.py              CPU replacements for CUDA-only FPS/gather (see below)
  pointattn_baseline.py   Experiment 1: original architecture, CPU-adapted
  pointattn_geo.py        Experiments 2-4: our geometry-enhanced encoder
utils/
  geometry_features.py    surface normal + curvature computation (PCA-based)
  normalization.py        normalize / denormalize, center+scale persistence
  metrics.py              Chamfer Distance, Precision, Recall, F1
train_baseline.py          <- training script for Experiment 1
train_contribution.py      <- training script for Experiments 2/3/4
tests/test_pipeline_smoke.py  end-to-end CPU smoke test on synthetic data
LICENSE_NOTE.md             attribution / licensing for adapted upstream code
```

## 9. Why the code differs from the upstream repo on CPU

The original PointAttN repo's point-sampling ops (`furthest_point_sample`,
`gather_points`) come from a **compiled CUDA extension**
(`utils/mm3d_pn2`), and its Chamfer-distance loss uses another CUDA
kernel. Both are GPU-only. Since this project trains on Google Colab
**CPU** (confirmed with the supervisor), `models/cpu_ops.py` reimplements
the two sampling ops in plain PyTorch, and `utils/metrics.py` uses the
upstream repo's *own* pure-Python Chamfer distance fallback
(`chamfer_python.py`) instead of the CUDA kernel. The model architecture
itself is byte-for-byte the same logic as upstream — only the compute
backend changed. This is documented in detail at the top of
`models/cpu_ops.py` and `models/pointattn_baseline.py`.

## 10. How to run

```bash
pip install -r requirements.txt

# 1) Smoke-test everything on synthetic data first (no real dataset needed):
python tests/test_pipeline_smoke.py

# 2) Prepare real data (after downloading/extracting MUG500+ PLY files):
python data/prepare_mug500.py --raw_dir /path/to/MUG500plus/complete_skulls \
                               --out_dir data/processed/mug500

# 3) Train the baseline (Experiment 1):
python train_baseline.py --config configs/exp1_original.yaml

# 4) Train the contribution (Experiments 2-4):
python train_contribution.py --config configs/exp2_normals.yaml
python train_contribution.py --config configs/exp3_curvature.yaml
python train_contribution.py --config configs/exp4_geometry_enhanced.yaml
```

## 11. Known limitations / what's still open

- **Not yet run on real data or a real PyTorch runtime.** This code was
  written and unit-tested piece-by-piece (data prep, normalization,
  geometry features all verified against synthetic cases with known
  correct answers — see each module's `__main__` block and
  `tests/test_pipeline_smoke.py`'s docstring for exactly what was and
  wasn't verified), but the environment that produced it had no PyTorch
  installed and no internet access, so **no PyTorch model has actually
  been executed yet.** Run `tests/test_pipeline_smoke.py` first in your
  Colab notebook to confirm the full pipeline runs before trusting any
  real training results.
- The exact folder layout of the real MUG500+ figshare archive hasn't been
  confirmed yet — `prepare_mug500.py --raw_dir` currently expects a flat
  directory of `.ply` files; adjust `_load_ply_points`/`main()` if the real
  archive nests files differently.
- PPCNet (VTK) loading/preprocessing is not yet implemented — this
  repository currently covers MUG500+ only; PPCNet needs its own loader
  (`.vtk` reading + the pre-defined 829/103/105 split) before Experiment
  generalization numbers can be produced.
- CloudCompare visual-comparison exports (Ground Truth / Incomplete Input /
  Completed Output triplets, report section 9) are not yet automated.
