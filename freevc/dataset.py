"""Parallel reconstruction dataset for the FreeVC reconstructor.

Training triplet (per VC sample):
- content: the VC sample wav -> WavLM content `c` (computed in the training step, or read
  from the WavLM cache)
- speaker: the source-speaker embedding `g` (from the exported npz)
- target:  the original LibriSpeech source utterance -> spec + waveform `y`

FreeVC needs `c` and the target frame-aligned, so both are cropped to the shorter length.
`align=True` first removes a constant VC/source offset (Train-1, Train-4), estimated from the
two waveforms' energy contours (`dataset.align`). The linear spectrogram and WavLM features
are computed on the GPU in the training loop.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.align import best_lag
from dataset.sltsstc_tar import read_wav

FRAME_HOP = 320  # 16 kHz / 50 fps: WavLM & FreeVC-spec frame stride, also the energy-contour hop


def _energy_contour(wav: np.ndarray, hop: int = FRAME_HOP, win: int = 400) -> np.ndarray:
    """Per-frame log-RMS energy of a 16 kHz waveform: a speaker-robust timing contour."""
    if len(wav) < win:
        return np.zeros(1, dtype=np.float32)
    n = 1 + (len(wav) - win) // hop
    frames = np.lib.stride_tricks.sliding_window_view(wav, win)[::hop][:n]
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(-1) + 1e-8)
    return np.log(rms + 1e-6).astype(np.float32)


def estimate_lag_samples(vc: np.ndarray, tgt: np.ndarray, max_lag_frames: int = 64) -> int:
    """Sample offset of `vc` vs `tgt` (a multiple of FRAME_HOP) that best aligns them.

    Positive => `vc` lags `tgt` (drop `vc`'s first N samples to align). Negative => drop
    `tgt`'s first |N|. Only used without the WavLM cache: the cache holds features of the
    unaligned audio, so `--align` needs WavLM to run live.
    """
    lag, _ = best_lag(_energy_contour(vc), _energy_contour(tgt), max_lag_frames)
    return lag * FRAME_HOP


class ParallelReconDataset(Dataset):
    """Returns `(content, target_wav, g)`.

    `content` is either the raw VC waveform (cropped to the target's length) or, with a
    `wavlm_cache` (`dataset.npz_cache.NpzCache`), the cached WavLM feature `(D, T)` of that
    VC sample. With a cache there is no waveform crop: the training step crops the feature and
    the target spectrogram to the shorter frame count.
    """

    def __init__(self, samples: list[tuple[str, str, np.ndarray]], sample_rate: int = 16000,
                 min_wav_len: int = 4000, align: bool = False, align_max_lag: int = 64,
                 wavlm_cache=None):
        self.samples = samples
        self.sample_rate = sample_rate
        self.min_wav_len = min_wav_len
        self.align = align            # raw-mode only (needs the VC waveform, no WavLM cache)
        self.align_max_lag = align_max_lag
        self.wavlm_cache = wavlm_cache  # NpzCache or None (set: content = cached WavLM feature)

    def __len__(self) -> int:
        return len(self.samples)

    def _load(self, path: str) -> np.ndarray:
        # from the tars when the corpus is packed, else librosa (float32 in [-1, 1])
        return read_wav(path, sr=self.sample_rate)

    def __getitem__(self, idx: int):
        content_path, tgt_path, g = self.samples[idx]
        g = torch.from_numpy(np.asarray(g, dtype=np.float32))
        tgt = self._load(tgt_path)

        if self.wavlm_cache is not None:  # cached WavLM feature (D, T) from <cache>/<subset>.npz
            c = self.wavlm_cache.get(content_path)
            c = c[0] if c.ndim == 3 else c  # tolerate a stored (1, D, T)
            if len(tgt) < self.min_wav_len:
                tgt = np.pad(tgt, (0, self.min_wav_len - len(tgt)))  # zero-pad very short targets
            return torch.from_numpy(np.ascontiguousarray(c)).float(), torch.from_numpy(tgt).float(), g

        vc = self._load(content_path)
        if self.align:  # correct a constant VC/source offset before the frame-0 crop
            lag = estimate_lag_samples(vc, tgt, self.align_max_lag)
            if lag > 0:
                vc = vc[lag:]
            elif lag < 0:
                tgt = tgt[-lag:]
        n = min(len(vc), len(tgt))  # then crop both to the shared length
        vc, tgt = vc[:n], tgt[:n]
        if n < self.min_wav_len:  # pad very short pairs
            pad = self.min_wav_len - n
            vc, tgt = np.pad(vc, (0, pad)), np.pad(tgt, (0, pad))
        return torch.from_numpy(vc).float(), torch.from_numpy(tgt).float(), g


def collate(batch):
    """Pad to batch max. Returns (content, tgt [B, L], lengths [B], g [B, D]).

    `content` is `[B, L]` raw waveforms or, for cached features, `[B, D, T]`
    (dispatched on item rank). `lengths` are target-waveform sample lengths.
    """
    contents, tgts, gs = zip(*batch)
    lengths = torch.tensor([t.size(0) for t in tgts], dtype=torch.long)
    max_len = int(lengths.max())
    tgt_p = torch.zeros(len(batch), max_len, dtype=torch.float32)
    for i, tgt in enumerate(tgts):
        tgt_p[i, : tgt.size(0)] = tgt

    if contents[0].dim() == 2:  # cached WavLM features (D, T) -> [B, D, T]
        D = contents[0].size(0)
        t_max = max(c.size(1) for c in contents)
        content_p = torch.zeros(len(batch), D, t_max, dtype=torch.float32)
        for i, c in enumerate(contents):
            content_p[i, :, : c.size(1)] = c
    else:  # raw VC waveforms (L,) -> [B, L] (same length as target per item)
        content_p = torch.zeros(len(batch), max_len, dtype=torch.float32)
        for i, c in enumerate(contents):
            content_p[i, : c.size(0)] = c

    return content_p, tgt_p, lengths, torch.stack(gs, dim=0)
