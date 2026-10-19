"""UTMOS22, the reference-free MOS predictor behind the naturalness tables in `analysis/`.

UTMOS22 (Saeki et al. 2022, tarepan/SpeechMOS `utmos22_strong`, the VoiceMOS 2022 winner) maps a
16 kHz waveform to a predicted 1-5 mean opinion score. It is calibrated for ranking systems, so
read per-system medians and means rather than single scores.

Offline compute nodes need the model in the torch.hub cache ($TORCH_HOME/hub) first, fetched once
on the frontend with `torch.hub.load("tarepan/SpeechMOS", "utmos22_strong")`.
"""

from __future__ import annotations

import numpy as np

_SR = 16000
_MIN = _SR // 20  # 50 ms, shorter clips give the SSL frontend no frames to pool, so they score NaN


def load_utmos22(device: str):
    """The UTMOS22 model in eval mode on `device`."""
    import torch

    return torch.hub.load("tarepan/SpeechMOS", "utmos22_strong", trust_repo=True).to(device).eval()


def _prep(wav) -> np.ndarray:
    """A numpy array or torch tensor as a flat float32 numpy waveform."""
    import torch

    if torch.is_tensor(wav):
        wav = wav.detach().cpu().numpy()
    return np.asarray(wav, dtype=np.float32).ravel()


def score_utmos22(model, wav, device: str) -> float:
    """Predicted MOS of one 16 kHz waveform, NaN if it is shorter than 50 ms."""
    import torch

    wav = _prep(wav)
    if wav.size < _MIN:
        return float("nan")
    with torch.no_grad():
        return float(model(torch.from_numpy(wav).unsqueeze(0).to(device), _SR))
