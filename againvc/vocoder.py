"""MelGAN vocoder backend for the AGAIN-VC reconstructor (mel -> waveform).

AGAIN-VC produces a mel. The waveform for ASV/EER scoring comes from the pretrained
universal MelGAN (`descriptinc/melgan-neurips`, loaded via torch.hub), exactly
as upstream AGAIN-VC (`util/vocoder.py`) does. The mel convention in `againvc/dsp.py`
is kept compatible with this vocoder (22.05 kHz, log10 mel). For compute nodes without
internet, fetch the hub repo once on a machine with access (it lands in TORCH_HOME's hub dir).
It is then loaded from there with `source="local"`, since a remote `torch.hub.load` hangs for
about 2 minutes on a dead network before falling back to the cache.

Loaded on first use and cached per device, so it takes no memory during training until the
first validation and is not part of any checkpoint.
"""

from __future__ import annotations

import os

import numpy as np
import torch

SAMPLE_RATE = 22050

_CACHE: dict = {}


def _load_melgan():
    """torch.hub-load the MelGAN vocoder, patching a librosa API incompatibility.

    descriptinc/melgan-neurips' `Audio2Mel.__init__` calls
    `librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax)` positionally, but modern librosa
    made `mel()` keyword-only (`TypeError: mel() takes 0 positional arguments`). We
    temporarily wrap `librosa.filters.mel` to accept the old positional signature just
    while the hub module imports/constructs (the `from librosa.filters import mel` binding
    inside the vendored code captures the shim), then restore it.
    """
    import librosa

    _orig = librosa.filters.mel

    def _mel_positional_compat(*args, **kwargs):
        if args:
            kwargs.update(dict(zip(["sr", "n_fft", "n_mels", "fmin", "fmax"], args)))
        return _orig(**kwargs)

    # Prefer the pre-fetched local repo (offline nodes): TORCH_HOME/hub/<owner>_<repo>_<ref>.
    # Fall back to a remote fetch only when it isn't cached (e.g. first fetch on the frontend).
    local = os.path.join(torch.hub.get_dir(), "descriptinc_melgan-neurips_master")
    librosa.filters.mel = _mel_positional_compat
    try:
        # trust_repo=True: skip torch.hub's interactive "trust this repo?" prompt, which
        # EOFErrors in non-interactive runs.
        if os.path.isdir(local):
            return torch.hub.load(local, "load_melgan", source="local", trust_repo=True)
        return torch.hub.load("descriptinc/melgan-neurips", "load_melgan", trust_repo=True)
    finally:
        librosa.filters.mel = _orig


def get_vocoder(device: str = "cuda"):
    """Lazy, per-device MelGAN singleton (descriptinc/melgan-neurips via torch.hub)."""
    if device not in _CACHE:
        # torch.hub's MelVocoder handles its own device: it moves the mel to that device
        # inside .inverse(...) and runs the generator there (GPU when available).
        _CACHE[device] = _load_melgan()
    return _CACHE[device]


def mel2wav(mel, device: str = "cuda") -> np.ndarray:
    """log10 mel (n_mels, T) or (1, n_mels, T), tensor/ndarray -> 1-D wav @22.05 kHz."""
    voc = get_vocoder(device)
    if not isinstance(mel, torch.Tensor):
        mel = torch.from_numpy(np.asarray(mel, dtype=np.float32))
    mel = mel.squeeze()
    mel = mel.unsqueeze(0).float()  # (1, n_mels, T)
    with torch.no_grad():
        wav = voc.inverse(mel).squeeze().detach().cpu().numpy()
    return wav
