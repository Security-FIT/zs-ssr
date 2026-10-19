"""Mel-spectrogram front-end for the DiffVC reconstructor.

Reproduces DiffVC's `get_mel` (from `inference.ipynb`) exactly, because the pretrained "average
voice" encoder and the universal HiFi-GAN vocoder were trained on this mel: 22.05 kHz, n_fft 1024,
hop 256, win 1024, 80 mels, fmin 0, fmax 8000, `center=False` with a 384-sample reflect pad,
magnitude STFT, and natural-log (not log10) compression clamped at 1e-5. Computed with
numpy/librosa like upstream, so the frozen encoder and the vocoder see identical features.
"""

from __future__ import annotations

import librosa
import numpy as np
from librosa.filters import mel as librosa_mel_fn

from dataset.sltsstc_tar import read_wav

SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 80
F_MIN = 0
F_MAX = 8000

_mel_basis = librosa_mel_fn(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX)


def load_wav(path: str) -> np.ndarray:
    """Load a mono waveform at 22.05 kHz (as DiffVC's get_mel does)."""
    return read_wav(path, sr=SAMPLE_RATE)  # from the tars when the corpus is packed, else librosa


def wav_to_mel(wav: np.ndarray) -> np.ndarray:
    """Waveform -> DiffVC log-mel, shape (n_mels, T). Byte-compatible with get_mel."""
    wav = wav[: (wav.shape[0] // HOP_LENGTH) * HOP_LENGTH]
    wav = np.pad(wav, 384, mode="reflect")
    stft = librosa.core.stft(wav, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
                             window="hann", center=False)
    stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + 1e-9)
    mel = np.matmul(_mel_basis, stftm)
    log_mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
    return log_mel.astype(np.float32)


def get_mel(wav_path: str) -> np.ndarray:
    """Convenience: load a wav @22.05 kHz and return its DiffVC log-mel (n_mels, T)."""
    return wav_to_mel(load_wav(wav_path))


def main():
    """Self-test: 1 s of noise gives a mel of the expected shape with finite values."""
    y = (np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.1)
    mel = wav_to_mel(y)
    assert mel.shape[0] == N_MELS and mel.ndim == 2, mel.shape
    assert np.isfinite(mel).all()
    print(f"[mel.py] wav_to_mel OK: {y.shape} -> {mel.shape} (natural-log mel, finite)")


if __name__ == "__main__":
    main()
