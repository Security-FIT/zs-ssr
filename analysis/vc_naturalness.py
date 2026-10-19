"""Naturalness of the VC samples per system: UTMOS22 predicted MOS, no reference needed.

Target-speaker similarity (vc_quality.py) alone does not rank the VC systems fairly. A retrieval
VC like kNN-VC (Test-8) builds its output from real target-speaker frames, so it scores far above
the generative systems in similarity although it sounds rougher. Naturalness covers that gap.

Every VC utterance is scored with UTMOS22 (`evaluation/utmos.py`), which maps a waveform to a
predicted 1-5 MOS. It is calibrated for ranking systems, so read the per-system median and mean.
The file list of each system comes from `analysis/results/vc_quality_Test-N.tsv`. A small
`--limit` runs on CPU, the full sets want a GPU. Offline nodes need the model cached first (see
scripts/sge/analysis/vc_naturalness.sh).

    python -m analysis.vc_naturalness --slt-root ../datasets/slt-sstc [--limit 1500]
        [--synths 2 5 8] [--out analysis/results/naturalness_table.tsv]
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np

from evaluation.utmos import load_utmos22, score_utmos22

COLS = ["n", "utmos22_median", "utmos22_mean", "utmos22_ci95_lo", "utmos22_ci95_hi",
        "utmos22_p2p5", "utmos22_p97p5"]


def summarize(scores: np.ndarray) -> dict:
    """Per-system statistics of the per-utterance UTMOS22 scores, keyed like COLS.

    Two 95% intervals: ci95 is the confidence interval of the mean (mean +/- 1.96 standard
    errors, narrow at n ~ 13k), p2p5..p97p5 holds the central 95% of the utterance scores.
    """
    mean = float(np.mean(scores))
    se = float(np.std(scores, ddof=1) / np.sqrt(len(scores)))  # standard error of the mean
    return {
        "n": len(scores),
        "utmos22_median": float(np.median(scores)),
        "utmos22_mean": mean,
        "utmos22_ci95_lo": mean - 1.96 * se,
        "utmos22_ci95_hi": mean + 1.96 * se,
        "utmos22_p2p5": float(np.percentile(scores, 2.5)),
        "utmos22_p97p5": float(np.percentile(scores, 97.5)),
    }


def synth_files(results_dir: str, synth: int) -> list[str]:
    """Dataset-relative VC wav paths of Test-`synth`, in vc_quality_Test-N.tsv order."""
    path = os.path.join(results_dir, f"vc_quality_Test-{synth}.tsv")
    with open(path) as f:
        # QUOTE_NONE because the transcripts in the TSV are not escaped, see vc_quality.py
        return [r["file_path"] for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)]


def measure(slt_root: str, results_dir: str, synths: list[int], limit: int, seed: int):
    """One summarize() row per system in `synths`, from up to `limit` random utterances each."""
    import torch

    from dataset import sltsstc_tar

    sltsstc_tar.activate(slt_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m_22 = load_utmos22(device)
    print(f"[vc_naturalness.py] device={device}")
    rng = np.random.default_rng(seed)
    rows = []
    for s in synths:
        files = synth_files(results_dir, s)
        if limit and limit < len(files):
            files = [files[i] for i in rng.choice(len(files), size=limit, replace=False)]
        u22 = []
        for fp in files:
            wav = sltsstc_tar.read_wav(os.path.join(slt_root, fp), sr=16000)
            b = score_utmos22(m_22, wav, device)
            if not np.isnan(b):  # NaN for clips too short to score
                u22.append(b)
        row = {"synth": s, **summarize(np.array(u22))}
        rows.append(row)
        print(f"[vc_naturalness.py] Test-{s}: n={row['n']}  UTMOS22 median={row['utmos22_median']:.2f} "
              f"mean={row['utmos22_mean']:.2f} "
              f"(95%CI {row['utmos22_ci95_lo']:.2f}-{row['utmos22_ci95_hi']:.2f})")
    return rows


def self_test():
    """Load UTMOS22 and score 3 s of noise and a too-short clip."""
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m_22 = load_utmos22(dev)
    b = score_utmos22(m_22, (0.1 * np.random.randn(3 * 16000)).astype(np.float32), dev)
    assert np.isfinite(b) and 1.0 <= b <= 5.0, b
    assert np.isnan(score_utmos22(m_22, np.zeros(10, np.float32), dev))
    s = summarize(np.array([1.0, 2.0, 3.0, 4.0]))
    assert s["n"] == 4 and s["utmos22_mean"] == 2.5 and s["utmos22_median"] == 2.5, s
    assert s["utmos22_ci95_lo"] < 2.5 < s["utmos22_ci95_hi"], s
    print(f"[vc_naturalness.py] self-test OK (device={dev}, UTMOS22 of noise={b:.2f})")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slt-root", default="../datasets/slt-sstc")
    p.add_argument("--results-dir", default="analysis/results")
    p.add_argument("--synths", nargs="+", type=int, default=list(range(1, 9)))
    p.add_argument("--limit", type=int, default=1500,
                   help="random VC utterances per system, 0 for all (a few hundred already give a stable median)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default=None, help="default <results-dir>/naturalness_table.tsv")
    p.add_argument("--self-test", action="store_true", help="run the self-test and exit")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return

    rows = measure(args.slt_root, args.results_dir, args.synths, args.limit, args.seed)
    out = args.out or os.path.join(args.results_dir, "naturalness_table.tsv")
    cols = ["synth"] + COLS
    with open(out, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n")
    print(f"\n[vc_naturalness.py] wrote {out}")


if __name__ == "__main__":
    main()
