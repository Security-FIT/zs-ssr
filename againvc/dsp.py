"""Mel-spectrogram front-end for the AGAIN-VC reconstructor.

Vendored from AGAIN-VC (https://github.com/KimythAnly/AGAIN-VC, MIT, see againvc/LICENSE),
`util/dsp.py`. The mel convention (22.05 kHz, n_fft 1024, hop 256, win 1024, 80 mels, fmin 0,
fmax 11025, log10 magnitude mel) must match the pretrained MelGAN vocoder
(descriptinc/melgan-neurips), so it keeps the numbers AGAIN-VC trained and vocoded with.

Two deliberate deviations from upstream `util/dsp.py`, each commented at its site:
1. `load_wav` does not trim leading/trailing silence: the reconstructor pairs a VC sample with
   its LibriSpeech original frame by frame, and trimming each independently would break that.
2. `wav2mel` clamps the mel magnitude to `>= 1e-5` before `log10` (upstream has no clamp), as
   descriptinc's Audio2Mel does, so silent bins can't give `log10(0) = -inf`.
"""

from __future__ import annotations

import librosa
import numpy as np

from dataset.sltsstc_tar import read_wav

SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 80
F_MIN = 0
F_MAX = 11025

_mel_basis = librosa.filters.mel(
    sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX
)


def load_wav(path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load a mono waveform at 22.05 kHz, clipped to [-1, 1].

    DEVIATION vs upstream util/dsp.py: no `librosa.effects.trim`, since the VC sample and its
    LibriSpeech target must stay frame-aligned.
    """
    y = read_wav(path, sr=sample_rate)  # from the tars when the corpus is packed, else librosa
    return np.clip(y, -1.0, 1.0)


def wav2mel(y: np.ndarray) -> np.ndarray:
    """Waveform -> log10 magnitude mel, shape (n_mels, T). Matches AGAIN-VC/MelGAN."""
    D = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH))
    S = np.dot(_mel_basis, D)
    # DEVIATION vs upstream: clamp before log10 (descriptinc Audio2Mel convention) so
    # empty mel bins can't produce -inf (numerically identical elsewhere).
    log_S = np.log10(np.maximum(S, 1e-5))
    return log_S.astype(np.float32)


def main():
    """Self-test: 1 s of noise gives a mel of the expected shape with finite values."""
    y = (np.random.randn(SAMPLE_RATE).astype(np.float32) * 0.1)
    mel = wav2mel(y)
    assert mel.shape[0] == N_MELS and mel.ndim == 2, mel.shape
    assert np.isfinite(mel).all()
    print(f"[dsp.py] wav2mel OK: {y.shape} -> {mel.shape} (log10 mel, finite)")


if __name__ == "__main__":
    main()
