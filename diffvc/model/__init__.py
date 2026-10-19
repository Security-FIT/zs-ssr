# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Vendored DiffVC model (https://github.com/huawei-noah/Speech-Backbones/tree/main/DiffVC,
MIT, see diffvc/LICENSE). Intra-package imports were relativized (`model.X` -> `.X`)
for packaging. Behavioral changes (external-`g` conditioning via `emb_dim`) are commented
inline in `diffusion.py` and `vc.py`.
"""

from .utils import fix_len_compatibility, sequence_mask
from .vc import DiffVC, FwdDiffusion

__all__ = ["DiffVC", "FwdDiffusion", "sequence_mask", "fix_len_compatibility"]
