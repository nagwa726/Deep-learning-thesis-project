"""
train_baseline.py
-------------------
Trains EXPERIMENT 1: the Original PointAttN model (models/pointattn_baseline.py),
unmodified from upstream except for the CPU-only sampling ops (see
models/cpu_ops.py) that let it run without a CUDA GPU.

This is the FIRST of the two scripts requested for the supervisor meeting:
it trains the plain baseline on the skull dataset and reports
Chamfer Distance / Precision / Recall / F1 exactly as section 8 of the
report specifies, so its numbers are directly comparable against
train_contribution.py's numbers for the ablation study (section 4).

Usage (run from the repo root, after prepare_mug500.py has produced
data/processed/mug500/):
    python train_baseline.py --config configs/exp1_original.yaml

Everything here runs on CPU (no .cuda() anywhere, unlike upstream's
train.py which hardcodes CUDA). Expect this to be slow -- that is exactly
why the report records "CPU only" as a known constraint (section 10).
"""

import argparse
import datetime
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from data.skull_dataset import SkullCompletionDataset, collate_skull_batch
from models.pointattn_baseline import Model
from utils.metrics import evaluate_batch
from utils.normalization import load_norm_params, denormalize


class AverageMeter:
    """Running mean tracker, same role as upstream's AverageValueMeter."""
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def denormalize_batch_to_mm(points_norm_batch: torch.Tensor, sample_ids: list[str], processed_dir: str) -> torch.Tensor:
    """
    Converts a batch of NORMALIZED point clouds back to physical units (mm),
    per-sample, using that sample's saved (center, scale) from
    prepare_mug500.py. Required before computing/reporting any metric in
    physical units -- see report section 6 ("Denormalization") and
    section 7 ("Physical Units"): metrics must not be reported in a unit
    before this step, and the per-sample center/scale (not a dataset-wide
    average) must be used so each skull is restored to ITS OWN original
    coordinate frame.
    """
    out = torch.empty_like(points_norm_batch)
    for b, sid in enumerate(sample_ids):
        norm_path = os.path.join(processed_dir, f"{sid}_norm.json")
        center, scale, _ = load_norm_params(norm_path)
        pts_np = points_norm_batch[b].detach().cpu().numpy()
        out[b] = torch.from_numpy(denormalize(pts_np, center, scale).astype(np.float32))
    return out


def run_validation(net, dataloader, cfg, epoch, logger):
    net.eval()
    meters = {k: AverageMeter() for k in ["cd_p", "cd_t", "precision", "recall", "f1"]}
    with torch.no_grad():
        for batch in dataloader:
            partial = batch["partial"]          # (B, 3, n_partial)  -- XYZ only, no geo feats
            complete = batch["complete"]        # (B, n_complete, 3)

            result = net(partial, complete, is_training=False)
            fine1 = result["out2"]              # (B, n_complete, 3), normalized space

            # Denormalize BOTH prediction and ground truth before metrics,
            # per report section 6: "يجب تطبيق التحويل نفسه على Completed
            # Point Cloud و Ground Truth قبل المقارنة."
            fine1_mm = denormalize_batch_to_mm(fine1, batch["sample_id"], cfg["processed_dir"])
            complete_mm = denormalize_batch_to_mm(complete, batch["sample_id"], cfg["processed_dir"])

            metrics = evaluate_batch(complete_mm, fine1_mm, threshold=cfg["f1_threshold_mm"])
            bsz = partial.shape[0]
            for k in meters:
                meters[k].update(metrics[k].mean().item(), n=bsz)

    logger.info(
        "[epoch %d][val] cd_p(mm)=%.4f cd_t(mm^2)=%.4f precision=%.4f recall=%.4f f1=%.4f",
        epoch, meters["cd_p"].avg, meters["cd_t"].avg,
        meters["precision"].avg, meters["recall"].avg, meters["f1"].avg,
    )
    return meters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="path to a configs/*.yaml file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["work_dir"], exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(cfg["work_dir"], "train.log")),
            logging.StreamHandler(sys.stdout),
        ],
        format="%(levelname)s - %(asctime)s - %(message)s",
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info("Config: %s", cfg)

    seed = cfg.get("manual_seed") or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    logger.info("Random seed: %d", seed)

    # ---- data ----
    # geo_mode="none": this is the plain baseline, so we never even load
    # normals/curvature, keeping the comparison with train_contribution.py
    # honest (no accidental information leakage into the baseline).
    train_ds = SkullCompletionDataset(cfg["processed_dir"], split="train", geo_mode="none")
    val_ds = SkullCompletionDataset(cfg["processed_dir"], split="val", geo_mode="none")
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=cfg["workers"], collate_fn=collate_skull_batch)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["workers"], collate_fn=collate_skull_batch)
    logger.info("train samples=%d, val samples=%d", len(train_ds), len(val_ds))

    # ---- model ----
    # `args` object below only needs a `.dataset` attribute because
    # upstream's Model.__init__ branches on it (kept for structural
    # fidelity with the original file); we don't use the pcn/c3d datasets
    # here, so the value is unused except to pick step1/step2, which our
    # baseline file already fixes explicitly -- see pointattn_baseline.py.
    class _Args:
        dataset = "pcn"
    net = Model(_Args())
    # NOTE: no net.cuda() -- this is the deliberate CPU adaptation.

    optimizer = optim.Adam(
        net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        betas=tuple(float(b) for b in cfg["betas"].split(",")),
    )

    train_loss_meter = AverageMeter()
    lr = cfg["lr"]

    for epoch in range(cfg["nepoch"]):
        net.train()
        train_loss_meter = AverageMeter()

        if cfg["lr_decay"] and epoch > 0 and epoch % cfg["lr_decay_interval"] == 0:
            lr = max(lr * cfg["lr_decay_rate"], cfg["lr_clip"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()

            partial = batch["partial"]      # (B, 3, n_partial)
            complete = batch["complete"]    # (B, n_complete, 3)

            _, loss2, net_loss = net(partial, complete, is_training=True)
            train_loss_meter.update(net_loss.item())

            net_loss.backward()
            optimizer.step()

            if i % cfg["step_interval_to_print"] == 0:
                logger.info(
                    "[epoch %d: %d/%d] fine_loss(normalized cd_t)=%.6f total_loss=%.6f lr=%.6g",
                    epoch, i, len(train_loader), loss2.mean().item(), net_loss.item(), lr,
                )

        if epoch % cfg["epoch_interval_to_save"] == 0:
            ckpt_path = os.path.join(cfg["work_dir"], "model_last.pth")
            torch.save({"net_state_dict": net.state_dict(), "epoch": epoch}, ckpt_path)
            logger.info("Saved checkpoint: %s", ckpt_path)

        if epoch % cfg["epoch_interval_to_val"] == 0 or epoch == cfg["nepoch"] - 1:
            run_validation(net, val_loader, cfg, epoch, logger)

    logger.info("Training finished (%s).", cfg["experiment_name"])


if __name__ == "__main__":
    main()
