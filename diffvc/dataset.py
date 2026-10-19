"""Parallel mel reconstruction dataset for the DiffVC reconstructor.

Training triplet (per VC sample):
- content prior: the VC sample -> DiffVC mel -> frozen encoder -> "average voice" mel
- speaker:       the source-speaker embedding `g` (from the exported npz)
- target:        the original LibriSpeech source utterance -> DiffVC mel (x0)

With `--libritts-cache` the frozen-encoder output is read from a per-subset npz cache
(`dataset.npz_cache`), so training skips the encoder forward, without it the encoder runs live
in the step on the raw VC mel. The target mel is always computed live, then both are cropped to
`train_frames`. `prime_encoder_cache` fills the cache on the first training run.

The per-frame score-matching loss needs content and target frame-aligned. Most synthesizers are
frame-synchronous (Train-3), some sit at a constant offset (Train-1, ~23 frames).
`align=True` removes it from the mels' energy contours (`dataset.align`) before cropping.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.align import best_lag
from dataset.npz_cache import subset_file_paths, write_subset
from diffvc.mel import load_wav, wav_to_mel
from diffvc.model.utils import sequence_mask


class ParallelMelReconDataset(Dataset):
    """Returns `(content, target_mel, length, g)` padded to `train_frames`.

    `content` is the cached avg-voice mel (when `mel_cache` is set) or the raw VC mel
    (computed from the wav). The caller must run the frozen encoder on the raw mel.
    """

    def __init__(self, samples: list[tuple[str, str, np.ndarray]], train_frames: int = 128,
                 align: bool = False, align_max_lag: int = 50, mel_cache=None):
        self.samples = samples
        self.train_frames = train_frames
        self.align = align
        self.align_max_lag = align_max_lag
        self.mel_cache = mel_cache      # dataset.npz_cache.NpzCache, or None (encoder runs live)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        content_path, tgt_path, g = self.samples[idx]
        g = torch.from_numpy(np.asarray(g, dtype=np.float32))
        t_mel = wav_to_mel(load_wav(tgt_path))                 # (80, Tt) LibriSpeech target
        if self.mel_cache is not None:
            content = self.mel_cache.get(content_path).astype(np.float32)  # cached avg-voice mel
        else:
            content = wav_to_mel(load_wav(content_path))        # raw VC mel (80, Tc)
        tf = self.train_frames
        if self.align:                                         # correct a constant VC/source frame offset
            lag, _ = best_lag(content.mean(0), t_mel.mean(0), self.align_max_lag)  # energy contours
            if lag > 0:
                content = content[:, lag:]
            elif lag < 0:
                t_mel = t_mel[:, -lag:]
        T = min(content.shape[1], t_mel.shape[1])              # then crop both to the shared length from frame 0
        content, t_mel = content[:, :T], t_mel[:, :T]
        if T >= tf:
            s = random.randint(0, T - tf)
            content, t_mel = content[:, s:s + tf], t_mel[:, s:s + tf]
            length = tf
        else:
            pad = tf - T
            content = np.pad(content, ((0, 0), (0, pad)))
            t_mel = np.pad(t_mel, ((0, 0), (0, pad)))
            length = T
        return torch.from_numpy(content).float(), torch.from_numpy(t_mel).float(), length, g


def collate(batch):
    """Stack -> (content [B,80,tf], target [B,80,tf], lengths [B], g [B,D])."""
    contents, tgts, lengths, gs = zip(*batch)
    return (torch.stack(contents, 0), torch.stack(tgts, 0),
            torch.tensor(lengths, dtype=torch.long), torch.stack(gs, 0))


@torch.no_grad()
def prime_encoder_cache(model, npz_paths, slt_root, cache_dir, device,
                        synthesizers=None, limit=None):
    """Precompute the frozen encoder's avg-voice mels into one `.npz` per VC subset
    (`dataset.npz_cache`). Skips subsets that are already cached.

    A subset npz is all-or-nothing: a cache primed with `limit` (smoke test) holds only the
    first `limit` utterances per subset, so delete it before a full run (or smoke-test with
    `--no-libritts-cache`).
    """
    model.encoder.eval()
    for subset, fps in subset_file_paths(npz_paths, synthesizers).items():
        if os.path.exists(os.path.join(cache_dir, subset + ".npz")):
            print(f"[dataset.py] {subset} already in the encoder cache, skip")
            continue
        fps = fps[:limit] if limit else fps

        def items(fps=fps, subset=subset):
            for fp in tqdm(fps, desc=f"libritts-cache {subset}"):
                mel = wav_to_mel(load_wav(os.path.join(slt_root, fp)))          # (80, T)
                x = torch.from_numpy(mel).float().unsqueeze(0).to(device)      # (1, 80, T)
                mask = sequence_mask(torch.LongTensor([mel.shape[1]]).to(device)).unsqueeze(1).to(x.dtype)
                avg = model.encoder(x, mask)                                   # (1, 80, T) avg-voice
                yield fp, avg.squeeze(0).cpu().numpy().astype(np.float32)

        write_subset(cache_dir, subset, items())
