"""AGAIN-VC reconstruction for the shared in-loop ASV validation.

`reconstruct(model, content_path, g, device, sr=...)` is the model-specific
reconstruction the shared harness (`evaluation/reconstruction_eval.py`,
`evaluation/examples.py`) calls: VC sample mel (content) + external source
embedding `g` -> AGAIN mel -> MelGAN waveform (22.05 kHz) -> resampled to the ASV
rate (`sr`, 16 kHz). AGAIN-VC has no content cache (its encoder is trained), so
`content_path` is always the VC wav.
"""

from __future__ import annotations

import librosa
import numpy as np
import torch

from againvc.dsp import SAMPLE_RATE, load_wav, wav2mel
from againvc.vocoder import mel2wav


@torch.inference_mode()
def reconstruct(model, content_path: str, g: np.ndarray, device: str, sr: int = 16000) -> np.ndarray:
    """VC sample + source `g` -> reconstructed 1-D waveform at `sr` (16 kHz for the ASV)."""
    mel = wav2mel(load_wav(content_path))                                   # (80, T) @22.05k
    mel_t = torch.from_numpy(mel).float().unsqueeze(0).to(device)          # (1, 80, T)
    g_t = torch.from_numpy(np.asarray(g, np.float32)).to(device).unsqueeze(0)  # (1, D)
    out_mel = model.inference(mel_t, g_t)                                   # (1, 80, T)
    wav = mel2wav(out_mel, device=device)                                   # 1-D @22.05k
    if sr != SAMPLE_RATE:                                                   # -> ASV rate (anti-aliased)
        wav = librosa.resample(wav, orig_sr=SAMPLE_RATE, target_sr=sr)
    return wav
