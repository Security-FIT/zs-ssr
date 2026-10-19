"""FreeVC-specific reconstruction for the in-loop ASV validation.

The shared ASV/EER harness (`evaluation/reconstruction_eval.py`) drives the
validation and calls a model-specific `reconstruct_fn(model, content_path, g,
device, sr=...) -> 1-D 16 kHz wav`. This file provides FreeVC's: WavLM content
features (from the VC wav, or the WavLM cache) + the external source embedding `g`
-> waveform via the FreeVC decoder, which is already 16 kHz end-to-end.
"""

from __future__ import annotations

import numpy as np
import torch

from dataset.sltsstc_tar import read_wav


@torch.inference_mode()
def reconstruct(model, content_path: str, g: np.ndarray, device: str, sr: int = 16000,
                wavlm_cache=None) -> np.ndarray:
    """Reconstruct source speech from a VC sample + source embedding g -> 1-D waveform.

    `content_path` is the VC wav. When a `wavlm_cache` (`NpzCache`) is given, the
    WavLM feature is read from `<cache>/<subset>.npz` (keyed by that path) instead of
    running WavLM live. FreeVC is end-to-end 16 kHz, so no resampling is needed. Bind
    the cache with `functools.partial(reconstruct, wavlm_cache=cache)` for the harness.
    """
    if wavlm_cache is not None:
        c = torch.from_numpy(np.ascontiguousarray(wavlm_cache.get(content_path))).float()
        c = c.unsqueeze(0) if c.dim() == 2 else c                     # (1, ssl_dim, T)
        c = c.to(device)
    else:
        wav = read_wav(content_path, sr=sr)  # from the tars when the corpus is packed
        x = torch.from_numpy(wav).float().unsqueeze(0).to(device)     # (1, L)
        c = model.extract_wavlm_features(x)                           # (1, ssl_dim, T)
    g_t = torch.from_numpy(np.asarray(g, np.float32)).to(device).unsqueeze(0).unsqueeze(-1)  # (1, gin, 1)
    audio = model.inference(c, g=g_t)                                  # (1, 1, T')
    return audio.squeeze().detach().cpu().float().numpy()
