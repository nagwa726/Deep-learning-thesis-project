"""
pointattn_geo.py
-----------------
OUR CONTRIBUTION: "Geometry-Enhanced PointAttN"

This file contains ONLY what differs from the baseline
(models/pointattn_baseline.py). Everything else (cross_transformer,
PCT_refine, the two-stage refinement, the loss) is imported unchanged from
the baseline file and reused as-is, so the comparison between Experiment 1
(Original) and Experiments 2-4 (+Normals / +Curvature / +Both) isolates
EXACTLY one thing: what goes into the encoder's first convolution.

WHERE THE MODIFICATION LIVES (report section 3, "مكان التعديل الأساسي"):
The report is explicit that the geometric features must be fused INSIDE the
Encoder's feature-extraction step, not appended late in the pipeline. We
therefore modify only `PCT_encoder.__init__`'s `conv1` layer and
`PCT_encoder.forward`'s very first line, where XYZ is first turned into a
64-dim per-point feature. Concretely:

    baseline:  conv1 = Conv1d(3, 64)                 input = xyz                (3 ch)
    ours:      conv1 = Conv1d(3 + geo_dim, 64)        input = [xyz ; geo_feats]  (3+geo_dim ch)

geo_dim is 3 for normals-only, 1 for curvature-only, or 4 for both --
this one class supports all three "+something" ablation rows (Experiments
2, 3, 4) through a constructor argument, so the SAME code (same layer
sizes philosophy, same training settings) is reused for every ablation
row, which is what the report asks for ("يجب تثبيت إعدادات التدريب
والتقييم قدر الإمكان لضمان عدالة المقارنة" -- training/eval settings must
be held fixed across experiments for a fair comparison).

WHAT DOES NOT CHANGE, AND WHY: the point-sampling steps inside the encoder
(`furthest_point_sample`, `gather_points`) still operate on the PLAIN XYZ
tensor, never on the XYZ+geometry tensor. Farthest-point sampling measures
Euclidean distance between points in 3D space; if we accidentally fed it a
7-channel tensor (xyz+normals+curvature), it would compute nonsense
"distances" mixing millimetres with unit normal components and a curvature
ratio. So we keep a dedicated `xyz` variable flowing through the whole
encoder for sampling/gathering, and a separate `geo_feats` tensor that is
concatenated with xyz ONLY at the point where the first learned feature is
computed (conv1). This is the "early fusion inside feature extraction, not
a late, unjustified concatenation" design the report asks for.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cpu_ops import furthest_point_sample, gather_points
from models.pointattn_baseline import cross_transformer, PCT_refine, calc_cd


class PCT_encoder_geo(nn.Module):
    """
    Geometry-Enhanced encoder. Identical to the baseline PCT_encoder except:
      1. conv1 accepts (3 + geo_dim) input channels instead of 3.
      2. forward() takes a separate `geo_feats` tensor and concatenates it
         with xyz BEFORE conv1, while keeping plain `xyz` for sampling.

    Args:
        channel:  same meaning as baseline (hidden width), default 64.
        geo_dim:  number of extra per-point channels being fused:
                    0 -> behaves exactly like the original encoder (for
                         sanity-checking that geo_dim=0 reproduces the
                         baseline numerically)
                    1 -> curvature only                 (Experiment 3)
                    3 -> surface normals only            (Experiment 2)
                    4 -> normals + curvature             (Experiment 4,
                                                           "Geometry-Enhanced")
    """

    def __init__(self, channel=64, geo_dim=4):
        super().__init__()
        self.channel = channel
        self.geo_dim = geo_dim

        # ---- THE MODIFICATION (everything else below is unchanged) ----
        self.conv1 = nn.Conv1d(3 + geo_dim, 64, kernel_size=1)
        # -----------------------------------------------------------------

        self.conv2 = nn.Conv1d(64, channel, kernel_size=1)

        self.sa1 = cross_transformer(channel, channel)
        self.sa1_1 = cross_transformer(channel * 2, channel * 2)
        self.sa2 = cross_transformer(channel * 2, channel * 2)
        self.sa2_1 = cross_transformer(channel * 4, channel * 4)
        self.sa3 = cross_transformer(channel * 4, channel * 4)
        self.sa3_1 = cross_transformer(channel * 8, channel * 8)

        self.relu = nn.GELU()

        self.sa0_d = cross_transformer(channel * 8, channel * 8)
        self.sa1_d = cross_transformer(channel * 8, channel * 8)
        self.sa2_d = cross_transformer(channel * 8, channel * 8)

        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel * 4, 64, kernel_size=1)
        self.ps = nn.ConvTranspose1d(channel * 8, channel, 128, bias=True)
        self.ps_refuse = nn.Conv1d(channel, channel * 8, kernel_size=1)
        self.ps_adj = nn.Conv1d(channel * 8, channel * 8, kernel_size=1)

    def forward(self, xyz: torch.Tensor, geo_feats: torch.Tensor | None):
        """
        xyz:       (B, 3, N) raw coordinates -- used for ALL sampling/gather
                   steps below, exactly like the baseline encoder.
        geo_feats: (B, geo_dim, N) per-point geometric descriptors
                   (surface normals and/or curvature), or None if geo_dim=0.
        """
        batch_size, _, N = xyz.size()

        # ---- EARLY FUSION: concatenate geometry with xyz before conv1 ----
        if self.geo_dim > 0:
            assert geo_feats is not None, "geo_feats required when geo_dim > 0"
            points_aug = torch.cat([xyz, geo_feats], dim=1)  # (B, 3+geo_dim, N)
        else:
            points_aug = xyz
        x = self.relu(self.conv1(points_aug))  # first learned feature already sees geometry
        x0 = self.conv2(x)
        # --------------------------------------------------------------------

        # From here on this is IDENTICAL to the baseline encoder, operating
        # on plain `xyz` for sampling (never on points_aug):
        points = xyz
        idx_0 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 4)
        x_g0 = gather_points(x0, idx_0)
        points = gather_points(points, idx_0)
        x1 = self.sa1(x_g0, x0).contiguous()
        x1 = torch.cat([x_g0, x1], dim=1)
        x1 = self.sa1_1(x1, x1).contiguous()

        idx_1 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 8)
        x_g1 = gather_points(x1, idx_1)
        points = gather_points(points, idx_1)
        x2 = self.sa2(x_g1, x1).contiguous()
        x2 = torch.cat([x_g1, x2], dim=1)
        x2 = self.sa2_1(x2, x2).contiguous()

        idx_2 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 16)
        x_g2 = gather_points(x2, idx_2)
        x3 = self.sa3(x_g2, x2).contiguous()
        x3 = torch.cat([x_g2, x3], dim=1)
        x3 = self.sa3_1(x3, x3).contiguous()

        x_g = F.adaptive_max_pool1d(x3, 1).view(batch_size, -1).unsqueeze(-1)
        x = self.relu(self.ps_adj(x_g))
        x = self.relu(self.ps(x))
        x = self.relu(self.ps_refuse(x))

        x0_d = self.sa0_d(x, x)
        x1_d = self.sa1_d(x0_d, x0_d)
        x2_d = self.sa2_d(x1_d, x1_d).reshape(batch_size, self.channel * 4, N // 8)

        fine = self.conv_out(self.relu(self.conv_out1(x2_d)))
        return x_g, fine


class ModelGeo(nn.Module):
    """
    Full Geometry-Enhanced model. Only the encoder changes vs. baseline's
    `Model`; PCT_refine/refine1 and the loss are reused unmodified from
    models/pointattn_baseline.py on purpose (see module docstring).

    `geo_dim` selects which ablation row this is:
        0 -> Experiment 1 (Original PointAttN)          [use baseline.Model instead in practice;
                                                           kept here only so this class is a strict
                                                           superset for debugging/sanity checks]
        3 -> Experiment 2 (PointAttN + Surface Normals)
        1 -> Experiment 3 (PointAttN + Curvature)
        4 -> Experiment 4 (Geometry-Enhanced PointAttN: Normals + Curvature)
    """

    def __init__(self, args, geo_dim=4):
        super().__init__()
        step1, step2 = 4, 8  # identical reasoning/values as baseline.Model

        self.geo_dim = geo_dim
        self.encoder = PCT_encoder_geo(geo_dim=geo_dim)
        self.refine = PCT_refine(ratio=step1)
        self.refine1 = PCT_refine(ratio=step2)

    def forward(self, xyz: torch.Tensor, geo_feats: torch.Tensor | None,
                gt: torch.Tensor | None = None, is_training: bool = True):
        """
        xyz:       (B, 3, N) partial input point cloud.
        geo_feats: (B, geo_dim, N) matching per-point normals/curvature for
                   the SAME partial input (computed in data prep, see
                   data/skull_dataset.py); None if geo_dim == 0.
        gt:        (B, M, 3) complete ground truth (training/eval only).
        """
        feat_g, coarse = self.encoder(xyz, geo_feats)

        new_x = torch.cat([xyz, coarse], dim=2)
        new_x = gather_points(new_x, furthest_point_sample(new_x.transpose(1, 2).contiguous(), 512))

        fine, feat_fine = self.refine(None, new_x, feat_g)
        fine1, feat_fine1 = self.refine1(feat_fine, fine, feat_g)

        coarse = coarse.transpose(1, 2).contiguous()
        fine = fine.transpose(1, 2).contiguous()
        fine1 = fine1.transpose(1, 2).contiguous()

        if is_training:
            loss3, _ = calc_cd(fine1, gt)
            gt_fine1 = gather_points(
                gt.transpose(1, 2).contiguous(),
                furthest_point_sample(gt, fine.shape[1])
            ).transpose(1, 2).contiguous()

            loss2, _ = calc_cd(fine, gt_fine1)
            gt_coarse = gather_points(
                gt_fine1.transpose(1, 2).contiguous(),
                furthest_point_sample(gt_fine1, coarse.shape[1])
            ).transpose(1, 2).contiguous()

            loss1, _ = calc_cd(coarse, gt_coarse)

            total_train_loss = loss1.mean() + loss2.mean() + loss3.mean()
            return fine, loss2, total_train_loss
        else:
            cd_p, cd_t = calc_cd(fine1, gt)
            cd_p_coarse, cd_t_coarse = calc_cd(coarse, gt)
            return {
                "out1": coarse, "out2": fine1,
                "cd_t_coarse": cd_t_coarse, "cd_p_coarse": cd_p_coarse,
                "cd_p": cd_p, "cd_t": cd_t,
            }


# Convenience: ablation-row name -> geo_dim, used by train_contribution.py
# and the four configs/exp*.yaml files so there is exactly ONE place that
# defines what each experiment name means.
ABLATION_GEO_DIM = {
    "original": 0,        # Experiment 1 (use baseline.Model, not this file, in practice)
    "normals": 3,          # Experiment 2
    "curvature": 1,        # Experiment 3
    "geometry_enhanced": 4,  # Experiment 4 (our full contribution)
}
