"""
skull_dataset.py
------------------
PyTorch Dataset over the .npz samples produced by prepare_mug500.py.
Deliberately thin: all the real work (region removal, normalization,
geometry features) already happened offline in prepare_mug500.py, so this
class just loads arrays and hands back tensors -- keeping the training loop
fast on CPU.

Returns, per sample:
    partial    (3, n_partial)   float32  -- network input, XYZ only
    geo_feats  (geo_dim, n_partial) float32 or None -- normals/curvature,
                                    sliced according to which ablation
                                    experiment is requested
    complete   (n_complete, 3)  float32  -- ground truth for the loss/metrics
    sample_id  str                      -- for bookkeeping / error messages

`geo_mode` controls which columns of the precomputed (normals, curvature)
are exposed, so the SAME cached .npz files serve all four ablation
experiments (report section 4) without recomputing anything:
    "none"      -> geo_feats = None                        (Experiment 1)
    "normals"   -> geo_feats = normals                       (Experiment 2)
    "curvature" -> geo_feats = curvature[:, None]             (Experiment 3)
    "both"      -> geo_feats = concat(normals, curvature)     (Experiment 4)
"""

from __future__ import annotations
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

GEO_MODE_DIM = {"none": 0, "normals": 3, "curvature": 1, "both": 4}


class SkullCompletionDataset(Dataset):
    def __init__(self, processed_dir: str, split: str, geo_mode: str = "both"):
        """
        Args:
            processed_dir: output dir of prepare_mug500.py (contains
                manifest.json + one <id>.npz / <id>_norm.json per sample).
            split: "train", "val", or "test" (must match a key in the
                manifest's "split" dict).
            geo_mode: one of "none"/"normals"/"curvature"/"both", see
                module docstring.
        """
        assert geo_mode in GEO_MODE_DIM, f"geo_mode must be one of {list(GEO_MODE_DIM)}"
        self.processed_dir = processed_dir
        self.geo_mode = geo_mode

        manifest_path = os.path.join(processed_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)
        if split not in manifest["split"]:
            raise ValueError(f"split '{split}' not in manifest splits {list(manifest['split'])}")
        self.sample_ids = manifest["split"][split]
        if len(self.sample_ids) == 0:
            raise ValueError(
                f"split '{split}' is empty in {manifest_path} -- dataset is too small for "
                f"this split (increase --self_test_n / check val_frac,test_frac)."
            )

    def __len__(self):
        return len(self.sample_ids)

    def _geo_feats(self, normals: np.ndarray, curvature: np.ndarray) -> np.ndarray | None:
        if self.geo_mode == "none":
            return None
        if self.geo_mode == "normals":
            return normals
        if self.geo_mode == "curvature":
            return curvature[:, None]
        if self.geo_mode == "both":
            return np.concatenate([normals, curvature[:, None]], axis=1)
        raise AssertionError("unreachable")

    def __getitem__(self, i: int):
        sid = self.sample_ids[i]
        npz_path = os.path.join(self.processed_dir, f"{sid}.npz")
        d = np.load(npz_path)

        partial = d["partial"].astype(np.float32)        # (n_partial, 3)
        complete = d["complete"].astype(np.float32)       # (n_complete, 3)
        geo = self._geo_feats(d["normals"], d["curvature"])

        partial_t = torch.from_numpy(partial).transpose(0, 1).contiguous()     # (3, n_partial)
        complete_t = torch.from_numpy(complete).contiguous()                    # (n_complete, 3)
        geo_t = None if geo is None else torch.from_numpy(
            geo.astype(np.float32)
        ).transpose(0, 1).contiguous()                                          # (geo_dim, n_partial)

        return {"sample_id": sid, "partial": partial_t, "geo_feats": geo_t, "complete": complete_t}


def collate_skull_batch(batch: list[dict]) -> dict:
    """Default collation works for partial/complete (all same fixed size
    per prepare_mug500.py), but geo_feats may be None (geo_mode='none'), so
    we handle that explicitly rather than relying on torch's default
    collate (which errors on None)."""
    sample_ids = [b["sample_id"] for b in batch]
    partial = torch.stack([b["partial"] for b in batch], dim=0)
    complete = torch.stack([b["complete"] for b in batch], dim=0)
    if batch[0]["geo_feats"] is None:
        geo_feats = None
    else:
        geo_feats = torch.stack([b["geo_feats"] for b in batch], dim=0)
    return {"sample_id": sample_ids, "partial": partial, "geo_feats": geo_feats, "complete": complete}
