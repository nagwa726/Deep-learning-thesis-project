"""
train_contribution.py
-----------------------
Trains EXPERIMENTS 2, 3, and 4: our contribution, "Geometry-Enhanced
PointAttN" (models/pointattn_geo.py), with the encoder modification that
fuses Surface Normals and/or Curvature into the very first feature-
extraction layer (see models/pointattn_geo.py's module docstring for the
full architectural justification).

This is the SECOND of the two scripts requested for the supervisor
meeting. It is almost line-for-line identical to train_baseline.py on
purpose (same training loop, same optimizer settings, same metrics,
same logging) -- the ONLY differences are marked "# [CONTRIBUTION]" below.
This mirrors structure is intentional: for an ablation study to mean
anything, Experiments 1-4 must differ ONLY in what the model receives as
input, not in how they are trained or evaluated.

Which experiment runs is controlled entirely by `geo_mode` in the config
file (configs/exp2_normals.yaml / exp3_curvature.yaml /
exp4_geometry_enhanced.yaml):
    geo_mode: normals    -> Experiment 2 (+Surface Normals)
    geo_mode: curvature  -> Experiment 3 (+Curvature)
    geo_mode: both       -> Experiment 4 (Geometry-Enhanced, full contribution)

Usage:
    python train_contribution.py --config configs/exp4_geometry_enhanced.yaml
"""

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from data.skull_dataset import SkullCompletionDataset, collate_skull_batch, GEO_MODE_DIM
from models.pointattn_geo import ModelGeo                                   # [CONTRIBUTION]
from utils.metrics import evaluate_batch
from utils.normalization import load_norm_params, denormalize


class AverageMeter:
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
    """Identical to train_baseline.py's helper of the same name -- see that
    file's docstring for why this must happen before computing metrics."""
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
            partial = batch["partial"]
            geo_feats = batch["geo_feats"]        # [CONTRIBUTION] may be None only if geo_mode=="none"
            complete = batch["complete"]

            result = net(partial, geo_feats, complete, is_training=False)   # [CONTRIBUTION] extra arg
            fine1 = result["out2"]

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
    parser.add_argument("-c", "--config", required=True,
                         help="configs/exp2_normals.yaml | exp3_curvature.yaml | exp4_geometry_enhanced.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    geo_mode = cfg["geo_mode"]                                              # [CONTRIBUTION]
    assert geo_mode in ("normals", "curvature", "both"), (
        f"train_contribution.py expects geo_mode in ('normals','curvature','both'), "
        f"got '{geo_mode}'. For geo_mode='none' use train_baseline.py instead, "
        f"so the plain-baseline run never imports geometry-feature code at all."
    )
    geo_dim = GEO_MODE_DIM[geo_mode]                                        # [CONTRIBUTION]

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
    logger.info("Ablation setting: geo_mode=%s -> geo_dim=%d", geo_mode, geo_dim)

    seed = cfg.get("manual_seed") or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    logger.info("Random seed: %d", seed)

    # ---- data ----
    train_ds = SkullCompletionDataset(cfg["processed_dir"], split="train", geo_mode=geo_mode)  # [CONTRIBUTION]
    val_ds = SkullCompletionDataset(cfg["processed_dir"], split="val", geo_mode=geo_mode)        # [CONTRIBUTION]
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=cfg["workers"], collate_fn=collate_skull_batch)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["workers"], collate_fn=collate_skull_batch)
    logger.info("train samples=%d, val samples=%d", len(train_ds), len(val_ds))

    # ---- model ----
    class _Args:
        dataset = "pcn"
    net = ModelGeo(_Args(), geo_dim=geo_dim)                                 # [CONTRIBUTION]

    optimizer = optim.Adam(
        net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        betas=tuple(float(b) for b in cfg["betas"].split(",")),
    )

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

            partial = batch["partial"]
            geo_feats = batch["geo_feats"]                                  # [CONTRIBUTION]
            complete = batch["complete"]

            _, loss2, net_loss = net(partial, geo_feats, complete, is_training=True)  # [CONTRIBUTION]
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
            torch.save({"net_state_dict": net.state_dict(), "epoch": epoch, "geo_mode": geo_mode}, ckpt_path)
            logger.info("Saved checkpoint: %s", ckpt_path)

        if epoch % cfg["epoch_interval_to_val"] == 0 or epoch == cfg["nepoch"] - 1:
            run_validation(net, val_loader, cfg, epoch, logger)

    logger.info("Training finished (%s).", cfg["experiment_name"])


if __name__ == "__main__":
    main()
