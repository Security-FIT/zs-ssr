"""Pair SLT-SSTC VC samples with their source-speaker embedding + LibriSpeech target.

Model-agnostic data pairing, driven by the exported npz protocol files (see
`embeddings/EMBEDDINGS.md`). Each row gives a VC `file_paths`, its `embeddings`
(source `g`) and `source_speaker`. The matching LibriSpeech target is resolved
from the filename's trailing `reader-chapter-utt` triple.

`build_parallel_samples` returns `(vc_wav_path, target_wav_path, g)` triples that
every reconstructor consumes. The per-model feature extraction (WavLM features, mels)
lives in each package's dataset.
"""

from __future__ import annotations

import os

import numpy as np
from tqdm import tqdm

from dataset.resolver import LibriSpeechResolver, source_utt_id


def _maybe_norm(v: np.ndarray, normalize: bool) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9) if normalize else v


def build_speaker_embedding_table(
    npz_paths: list[str], normalize: bool = True, synthesizers: set[str] | None = None
) -> dict[str, np.ndarray]:
    """Per-`source_speaker` MEAN embedding -> {speaker: (D,)} (merged over npz files).

    A stable per-speaker vector (smooths out noisy embeddings, but can wash out
    identity). `normalize=True` L2-normalizes. Only `train_embeddings.npz` carries
    `source_speaker`. `synthesizers` (a set of names like {"Train-3"}) restricts the
    mean to those subsets, matching the data actually trained on.
    """
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for npz_path in npz_paths:
        d = np.load(npz_path, allow_pickle=True)
        assert "source_speaker" in d, f"{npz_path} has no source_speaker (use a train_embeddings.npz)"
        syn = d["synthesizer"] if "synthesizer" in d.files else [None] * len(d["embeddings"])
        for s, e, sy in zip(d["source_speaker"], d["embeddings"], syn):
            if synthesizers is not None and str(sy) not in synthesizers:
                continue
            s = str(s)
            e = np.asarray(e, dtype=np.float64)
            sums[s] = sums[s] + e if s in sums else e
            counts[s] = counts.get(s, 0) + 1
    return {s: _maybe_norm(sums[s] / counts[s], normalize).astype(np.float32) for s in sums}


def build_parallel_samples(
    npz_paths: list[str],
    slt_root: str,
    librispeech_root: str,
    average: bool = True,
    normalize: bool = True,
    limit: int | None = None,
    synthesizers: list[str] | None = None,
) -> list[tuple[str, str, np.ndarray]]:
    """Build (vc_wav, target_wav, g) triples from the exported npz.

    `vc_wav` is `<slt_root>/<file_path>`. `g` is the per-speaker mean (`average=True`) or the
    per-utterance embedding of that VC sample (`average=False`). `synthesizers`
    (e.g. `["Train-3"]`) keeps only rows from those VC subsets (the npz `synthesizer`
    field), or None for all subsets. Every VC sample is expected to have a LibriSpeech
    original, and a missing one is a hard error (no silent skipping). `limit` caps the
    number of triples (for quick smoke tests).
    """
    resolver = LibriSpeechResolver(librispeech_root)
    synth_filter = set(synthesizers) if synthesizers else None
    spk_table = (
        build_speaker_embedding_table(npz_paths, normalize=normalize, synthesizers=synth_filter)
        if average else None
    )

    samples: list[tuple[str, str, np.ndarray]] = []
    for npz_path in npz_paths:
        d = np.load(npz_path, allow_pickle=True)
        fps, embs = d["file_paths"], d["embeddings"]
        assert "source_speaker" in d
        spks = d["source_speaker"]
        synths = d["synthesizer"] if "synthesizer" in d.files else [None] * len(fps)
        for fp, e, s, syn in tqdm(zip(fps, embs, spks, synths), total=len(fps), desc=f"resolve {os.path.basename(npz_path)}"):
            if synth_filter is not None and str(syn) not in synth_filter:
                continue
            utt = source_utt_id(str(fp))
            target = resolver.path(utt)  # grab the path to the original source utterance
            if target is None:
                raise FileNotFoundError(
                    f"No LibriSpeech original for VC sample '{fp}' (utt '{utt}') under "
                    f"{librispeech_root} (subsets={resolver.subsets})"
                )
            g = spk_table[str(s)] if average else _maybe_norm(np.asarray(e, dtype=np.float64), normalize).astype(np.float32)
            samples.append((os.path.join(slt_root, str(fp)), target, g))
            if limit is not None and len(samples) >= limit:
                resolver.save_cache()
                return samples
    resolver.save_cache()
    return samples
