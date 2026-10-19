"""DiffVC reconstruction for the shared in-loop ASV validation.

`reconstruct(model, content_path, g, device, sr=...)` is the model-specific
reconstruction the shared harness (`evaluation/reconstruction_eval.py`,
`evaluation/examples.py`) calls: VC sample -> DiffVC mel -> frozen encoder
(inside `model.reconstruct`) -> diffusion reverse sampling conditioned on external
`g` -> HiFi-GAN waveform (22.05 kHz) -> resampled to the ASV rate (`sr`, 16 kHz).

`content_path` is the VC wav: validation runs the encoder live on its small subset, the
libritts cache only speeds up training. `n_timesteps` / `mode` are the diffusion sampling steps
and solver, which `train_diffvc.py` binds with functools.partial.
"""

from __future__ import annotations

import librosa
import numpy as np
import torch

from diffvc.mel import SAMPLE_RATE, get_mel
from diffvc.vocoder import mel2wav


@torch.no_grad()
def reconstruct(model, content_path: str, g: np.ndarray, device: str, sr: int = 16000,
                n_timesteps: int = 6, mode: str = "ml") -> np.ndarray:
    """VC sample + source `g` -> reconstructed 1-D waveform at `sr` (16 kHz for the ASV)."""
    mel = get_mel(content_path)                                             # (80, T) @22.05k
    x = torch.from_numpy(mel).float().unsqueeze(0).to(device)              # (1, 80, T)
    lengths = torch.LongTensor([mel.shape[1]]).to(device)
    g_t = torch.from_numpy(np.asarray(g, np.float32)).unsqueeze(0).to(device)  # (1, D)
    out_mel = model.reconstruct(x, lengths, g_t, n_timesteps=n_timesteps, mode=mode)  # (1, 80, T)
    wav = mel2wav(out_mel, device=device)                                  # 1-D @22.05k
    if sr != SAMPLE_RATE:                                                  # -> ASV rate (anti-aliased)
        wav = librosa.resample(wav, orig_sr=SAMPLE_RATE, target_sr=sr)
    return wav
