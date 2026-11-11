# Attribution & Licensing Notes

This repository builds on the **PointAttN** baseline:

> Wang, J., Cui, Y., Guo, D., Li, J., Liu, Q., Shen, C. (2024).
> "PointAttN: You Only Need Attention for Point Cloud Completion."
> *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(6), 5472-5480.
> DOI: [10.1609/aaai.v38i6.28356](https://doi.org/10.1609/aaai.v38i6.28356)
> Source code: https://github.com/ohhhyeahhh/PointAttN

`models/pointattn_baseline.py` is a direct adaptation of that repository's
`models/PointAttN.py`, with one deliberate change: the CUDA-only point
sampling operations (`furthest_point_sample`, `gather_points`, originally
from `utils/mm3d_pn2`) are replaced with the pure-PyTorch, CPU-compatible
implementations in `models/cpu_ops.py`, since this project trains on CPU
(Google Colab, no GPU). The network architecture itself is unchanged.

`utils/metrics.py`'s Chamfer-distance implementation is adapted from the
upstream repo's own `utils/ChamferDistancePytorch/chamfer_python.py`
(a CPU-only reference implementation already included there, as opposed to
its CUDA-compiled counterpart), and the Precision/Recall/F1 formulation is
adapted from that repo's `utils/ChamferDistancePytorch/fscore.py`.

At the time of writing, the upstream repository did not display an explicit
license file. **Before pushing any adapted code publicly, or submitting the
thesis, double-check the upstream repository for a LICENSE file and follow
its terms** (the project report itself flags this requirement in section
11: "كذلك يجب مراعاة ترخيص الكود الأصلي قبل إعادة توزيع أي ملفات منه"). If
no license is present, contacting the original authors before wide
redistribution is the safest approach; at minimum, this attribution notice
and inline citations in the code (see the top of each adapted file) should
be kept intact.

`utils/ChamferDistancePytorch` (vendored inside the upstream repo) itself
credits:
- ChamferDistancePytorch: https://github.com/ThibaultGROUEIX/ChamferDistancePytorch
- mm3d_pn2 (not used here -- CUDA only): https://github.com/Colin97/MSN-Point-Cloud-Completion

## Data

The MUG500+ and PPCNet datasets are **not** included in or redistributed by
this repository (see `.gitignore`). Follow each dataset's own access terms
to obtain them.
