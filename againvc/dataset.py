"""Parallel mel reconstruction dataset for the AGAIN-VC reconstructor.

Training triplet (per VC sample):
- content: the VC sample wav -> AGAIN log10 mel
- speaker: the source-speaker embedding `g` (from the exported npz)
- target:  the original LibriSpeech source utterance -> AGAIN log10 mel

Loads both waveforms at 22.05 kHz (pairing comes from `dataset.pairing`), computes the AGAIN
mels (`againvc/dsp.py`), frame-aligns them and random-crops a `seglen`-frame segment.
The L1 loss needs content and target frame-aligned, so `align=True` first removes a constant
VC/source offset (Train-1, Train-4), estimated from the mels' energy contours (`dataset.align`).
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import Dataset

from againvc.dsp import load_wav, wav2mel
from dataset.align import best_lag


class ParallelMelReconDataset(Dataset):
    """Returns `(content_mel, target_mel, g)`, both mels `(n_mels, seglen)`."""

    def __init__(self, samples: list[tuple[str, str, np.ndarray]], seglen: int = 128,
                 align: bool = False, align_max_lag: int = 50):
        self.samples = samples
        self.seglen = seglen
        self.align = align
        self.align_max_lag = align_max_lag

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        content_path, tgt_path, g = self.samples[idx]
        g = torch.from_numpy(np.asarray(g, dtype=np.float32))
        c_mel = wav2mel(load_wav(content_path))   # (80, Tc)
        t_mel = wav2mel(load_wav(tgt_path))       # (80, Tt)
        if self.align:                            # correct a constant VC/source frame offset
            lag, _ = best_lag(c_mel.mean(0), t_mel.mean(0), self.align_max_lag)  # energy contours
            if lag > 0:
                c_mel = c_mel[:, lag:]
            elif lag < 0:
                t_mel = t_mel[:, -lag:]
        T = min(c_mel.shape[1], t_mel.shape[1])   # then crop both from frame 0 to shared length
        c_mel, t_mel = c_mel[:, :T], t_mel[:, :T]
        if T >= self.seglen:                      # random crop a fixed segment
            s = random.randint(0, T - self.seglen)
            c_mel = c_mel[:, s:s + self.seglen]
            t_mel = t_mel[:, s:s + self.seglen]
        else:                                     # pad short pairs to seglen
            pad = self.seglen - T
            c_mel = np.pad(c_mel, ((0, 0), (0, pad)))
            t_mel = np.pad(t_mel, ((0, 0), (0, pad)))
        return torch.from_numpy(c_mel).float(), torch.from_numpy(t_mel).float(), g


def collate(batch):
    """Stack fixed-size items -> (content [B, n_mels, seglen], target [...], g [B, D])."""
    contents, tgts, gs = zip(*batch)
    return torch.stack(contents, 0), torch.stack(tgts, 0), torch.stack(gs, 0)
