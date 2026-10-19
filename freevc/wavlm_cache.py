"""WavLM content-feature cache for the FreeVC reconstructor.

WavLM is frozen, so its features for a VC sample never change. `train_freevc.py --wavlm-cache`
extracts them once per VC subset into a per-subset `.npz` (`dataset.npz_cache`) and training
then reads them instead of running WavLM every step (~2x faster steps, and WavLM never sits in
training VRAM). Features are `(ssl_dim, T)` float32.
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from dataset.npz_cache import write_subset


def generate_subset(cache_dir: str, subset: str, fps: list[str], load_wav, ssl_backend, device: str) -> str | None:
    """Extract WavLM features for `fps` and stream them into `<cache_dir>/<subset>.npz`.

    `load_wav(fp)` returns the 1-D 16 kHz VC waveform, `ssl_backend` is a `freevc.ssl.SSLBackend`
    on `device`. Returns the written path, or None when the subset is already cached.
    """

    def items():
        for fp in tqdm(fps, desc=f"wavlm-cache {subset}"):
            x = torch.from_numpy(load_wav(fp)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                c = ssl_backend.extract(x)  # (1, ssl_dim, T)
            yield fp, c.squeeze(0).cpu().numpy().astype(np.float32)

    return write_subset(cache_dir, subset, items())
