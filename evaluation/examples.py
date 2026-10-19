"""Save best / average / worst reconstruction examples for a listening test.

Reconstructs a random sample of held-out VC utterances, scores each reconstruction against
the real source speaker with the ECAPA-TDNN ASV (cosine to the paired LibriSpeech
original), then saves the `n` best, `n` around-median and `n` worst as wavs under
`<out_dir>/examples/` (VC input, reconstruction, real source reference each), plus a
`manifest.json` that lists them with their scores.
Called by every `train_<model>.py` once, after the final test eval.

`reconstruct_fn(model, vc_wav, g, device, sr=...) -> 1-D 16 kHz wav` is supplied by the
caller. The ranking is over the sampled candidates only, so best and worst are relative to
that sample.
"""

from __future__ import annotations

import json
import os

import librosa
import numpy as np
from tqdm import tqdm

from dataset.sltsstc_tar import read_wav


def _bands(order: list[int], n: int) -> dict[str, list[int]]:
    """Best, around-median and worst `n` indices of a list sorted by ascending score."""
    n = min(n, len(order))
    mid = len(order) // 2
    lo = max(0, mid - n // 2)
    return {
        "best": list(reversed(order[-n:])),   # highest score first
        "average": order[lo:lo + n],          # around the median
        "worst": order[:n],                   # lowest score first
    }


def generate_examples(model, asv, eval_set, device, out_dir, *, meta, reconstruct_fn,
                      n_per_category: int = 3, n_candidates: int = 200,
                      seed: int = 1234, sr: int = 16000) -> dict:
    """Reconstruct and score `n_candidates` utterances of `eval_set`, save the best, average and
    worst `n_per_category` with a manifest. `meta` (architecture, split, checkpoint, ...) is
    copied into the manifest, which is also returned.
    """
    import soundfile as sf  # lazy: only needed to write the example wavs

    rng = np.random.default_rng(seed)
    rows = eval_set.rows
    k = min(n_candidates, len(rows))
    idx = rng.choice(len(rows), size=k, replace=False)
    print(f"[examples.py] reconstructing and scoring {k} of {len(rows)} utterances "
          f"(best and worst are within this sample)")

    recons, scores = [], []
    for i in tqdm(idx, desc="examples"):
        r = rows[int(i)]
        wav = reconstruct_fn(model, r["vc_file"], r["g"], device, sr=sr)   # 1-D wav @ sr
        s = asv.cosine(asv.embed_wav(wav, sr), asv.embed_file(r["original"]))  # vs its own original
        recons.append(wav)
        scores.append(float(s))

    order = sorted(range(k), key=lambda j: scores[j])   # ascending: worst .. best
    ex_dir = os.path.join(out_dir, "examples")
    os.makedirs(ex_dir, exist_ok=True)

    examples = []
    for cat, band in _bands(order, n_per_category).items():
        for rank, j in enumerate(band, 1):
            r = rows[int(idx[j])]
            stem = f"{cat}{rank}_cos{scores[j]:.3f}"
            vc = read_wav(r["vc_file"], sr=sr)                       # model input (loose or tar corpus)
            src = librosa.load(r["original"], sr=sr, mono=True)[0]   # real source speaker (LibriSpeech)
            sf.write(os.path.join(ex_dir, stem + "_1vcinput.wav"), vc, sr)
            sf.write(os.path.join(ex_dir, stem + "_2recon.wav"), recons[j], sr)
            sf.write(os.path.join(ex_dir, stem + "_3source.wav"), src, sr)
            examples.append({
                "category": cat, "rank": rank, "score": round(scores[j], 4),
                "source_speaker": r["speaker"],
                "vc_file": r["vc_file"], "source_file": r["original"],
                "vcinput_wav": stem + "_1vcinput.wav",
                "recon_wav": stem + "_2recon.wav",
                "source_wav": stem + "_3source.wav",
            })

    manifest = {
        "metric": "cosine(ASV(recon), ASV(source original)); higher = better",
        "n_candidates": k,
        "n_per_category": n_per_category,
        **meta,
        "examples": examples,
    }
    with open(os.path.join(ex_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[examples.py] wrote {len(examples)} examples "
          f"({n_per_category} each of best, average, worst) to {ex_dir}")
    return manifest


def main():
    """Self-test of the band selection."""
    order = list(range(10))  # indices 0..9 in ascending score order
    b = _bands(order, 3)
    # worst from the bottom, best from the top (best first), average around index 5
    assert b["worst"] == [0, 1, 2] and b["best"] == [9, 8, 7] and b["average"] == [4, 5, 6], b
    print("[examples.py] self-test OK")


if __name__ == "__main__":
    main()
