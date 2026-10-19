"""Linear-spectrogram / mel helpers for FreeVC training.

Vendored from FreeVC (https://github.com/OlaWod/FreeVC, MIT, (c) 2021 Jingyi Li,
see freevc/LICENSE) and modified for PyTorch 2.x: the original used
``torch.stft(..., return_complex=False)`` which is removed in torch 2.x, so the
STFTs below use ``return_complex=True`` + ``view_as_real`` (numerically equivalent
magnitude). The shared STFT is factored into `_stft_mag`, and the out-of-range input prints
and the unused decompression helpers are dropped. Coqui's bundled `mel_processing` only has
`mel_spectrogram_torch`, so all three functions are kept here.
"""

import torch
from librosa.filters import mel as librosa_mel_fn

mel_basis: dict = {}
hann_window: dict = {}


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)


def _stft_mag(y, n_fft, hop_size, win_size, center=False):
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    ).squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,  # torch 2.x (original FreeVC used return_complex=False)
    )
    spec = torch.view_as_real(spec)
    return torch.sqrt(spec.pow(2).sum(-1) + 1e-6)


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    """Linear-magnitude spectrogram. Returns (B, n_fft//2 + 1, T)."""
    return _stft_mag(y, n_fft, hop_size, win_size, center=center)


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    """Project a linear spectrogram onto the mel basis (log-compressed)."""
    fmax_dtype_device = f"{fmax}_{spec.dtype}_{spec.device}"
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    return spectral_normalize_torch(torch.matmul(mel_basis[fmax_dtype_device], spec))


def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    """Log-mel spectrogram straight from a waveform. Returns (B, num_mels, T)."""
    mag = _stft_mag(y, n_fft, hop_size, win_size, center=center)
    fmax_dtype_device = f"{fmax}_{y.dtype}_{y.device}"
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
    return spectral_normalize_torch(torch.matmul(mel_basis[fmax_dtype_device], mag))
