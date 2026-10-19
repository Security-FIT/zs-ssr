"""HiFi-GAN vocoder backend for the DiffVC reconstructor (mel -> waveform).

Loads the universal HiFi-GAN generator (vendored `diffvc/hifigan.py` + the downloaded
`generator` checkpoint, config `diffvc/hifigan_config.json`) lazily and caches it, so it
never touches training/VRAM until the first validation and is not part of any trained
checkpoint.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from diffvc.hifigan import load_vocoder

SAMPLE_RATE = 22050
_HERE = os.path.dirname(__file__)
DEFAULT_CONFIG = os.path.join(_HERE, "hifigan_config.json")
DEFAULT_CKPT = os.path.join(_HERE, "checkpts", "vocoder", "generator")

_CACHE: dict = {}


def get_vocoder(ckpt: str = DEFAULT_CKPT, config: str = DEFAULT_CONFIG, device: str = "cuda"):
    """Lazy, per-(ckpt, device) HiFi-GAN generator singleton."""
    key = (ckpt, device)
    if key not in _CACHE:
        _CACHE[key] = load_vocoder(ckpt, config, device=device)
    return _CACHE[key]


def mel2wav(mel, device: str = "cuda", ckpt: str = DEFAULT_CKPT, config: str = DEFAULT_CONFIG) -> np.ndarray:
    """DiffVC log-mel (n_mels, T) or (1, n_mels, T) -> 1-D wav @22.05 kHz."""
    g = get_vocoder(ckpt, config, device)
    if not isinstance(mel, torch.Tensor):
        mel = torch.from_numpy(np.asarray(mel, dtype=np.float32))
    mel = mel.squeeze().unsqueeze(0).to(device).float()  # (1, n_mels, T)
    with torch.no_grad():
        wav = g(mel).squeeze().detach().cpu().numpy()
    return wav
