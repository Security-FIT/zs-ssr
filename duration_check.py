"""Check, per synthesizer, that a VC sample is frame-aligned with its LibriSpeech original.

All four reconstructors train on (VC sample, LibriSpeech original) pairs with a per-frame
loss and crop both from frame 0, which assumes the VC system keeps the source timing. If a
synthesizer shifts or stretches time, the input and target frames hold different phonemes and
training degrades silently.

For every synthesizer this measures:
  1. duration: frame-count ratio and |frame difference| (soundfile headers only, no decode)
  2. alignment: on a sub-sample, the lag at which the two mels' log-energy contours correlate
     best (`dataset.align.best_lag`). Lag ~0 with high correlation means aligned.

    python duration_check.py --per-synth 500 --xcorr 50
    python duration_check.py --synth Train-3 Train-5 --per-synth 500

Result on Train-1..8: duration is preserved everywhere (ratio ~1.0). Most synthesizers are
frame-synchronous (e.g. Train-3, lag ~0), but Train-1 (~23-frame lead, ~270 ms) and Train-4
(~255 ms) are shifted by a constant offset, hence the trainers' `--align` flag for those two.
"""

from __future__ import annotations

import argparse
import os
import statistics as stats

import numpy as np
import soundfile as sf

from dataset.align import best_lag
from dataset.resolver import LibriSpeechResolver, source_utt_id
from diffvc.mel import HOP_LENGTH, SAMPLE_RATE, get_mel

DEFAULT_NPZ = "embeddings/non-finetuned/organizer-pretrained/exported_embeddings/train_embeddings.npz"


def duration_frames(path: str) -> float:
    """Number of DiffVC mel frames for `path` from its header (no audio decode)."""
    info = sf.info(path)
    seconds = info.frames / info.samplerate
    return seconds * SAMPLE_RATE / HOP_LENGTH


def contour(path: str) -> np.ndarray:
    """Speaker-robust timing contour = per-frame mean of the DiffVC log-mel."""
    return get_mel(path).mean(axis=0)


def _med(xs: list[float]) -> float:
    """Median, NaN for an empty list."""
    return stats.median(xs) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", nargs="+", default=[DEFAULT_NPZ], help="exported *_embeddings.npz protocol file(s)")
    ap.add_argument("--slt-root", default="../datasets/slt-sstc", help="SLT-SSTC dataset root (holds the VC file_paths)")
    ap.add_argument("--librispeech-root", default="../datasets/LibriSpeech", help="LibriSpeech root (reconstruction targets)")
    ap.add_argument("--synth", nargs="*", default=None, help="restrict to these synthesizers (e.g. Train-3 Train-5); default all")
    ap.add_argument("--per-synth", type=int, default=500, help="rows sampled per synthesizer (evenly spaced across its block) for the duration check")
    ap.add_argument("--xcorr", type=int, default=50, help="pairs per synthesizer for the mel cross-correlation; 0 to skip")
    ap.add_argument("--max-lag", type=int, default=50, help="cross-correlation search window in frames (+/-)")
    args = ap.parse_args()

    resolver = LibriSpeechResolver(args.librispeech_root)
    synth_filter = set(args.synth) if args.synth else None

    # up to --per-synth rows per synthesizer, evenly spaced over its files (sorted by name, so
    # the first rows would all share a few target voices)
    picks: list[tuple[str, str]] = []  # (vc_file_path, synth)
    for npz_path in args.npz:
        d = np.load(npz_path, allow_pickle=True)
        fps = d["file_paths"]
        synths = d["synthesizer"].astype(str)
        for syn in sorted(set(synths.tolist())):
            if synth_filter is not None and syn not in synth_filter:
                continue
            idxs = np.nonzero(synths == syn)[0]
            if len(idxs) > args.per_synth:
                idxs = idxs[np.linspace(0, len(idxs) - 1, args.per_synth).astype(int)]
            picks += [(str(fps[i]), syn) for i in idxs]

    # (vc_path, target_path, synth) pairs: resolve the LibriSpeech original for each pick.
    pairs: list[tuple[str, str, str]] = []
    missing = 0
    for fp, syn in picks:
        tgt = resolver.path(source_utt_id(fp))
        if tgt is None:
            missing += 1
            continue
        pairs.append((os.path.join(args.slt_root, fp), tgt, syn))

    if not pairs:
        raise SystemExit("[duration_check.py] no pairs found, check --npz / --slt-root / --librispeech-root / --synth")
    print(f"[duration_check.py] loaded {len(pairs)} pairs ({missing} VC samples had no LibriSpeech original and were skipped)\n")

    # 1) Duration (header-only): frame ratio + |frame diff| per synthesizer.
    dur: dict[str, list[tuple[float, float]]] = {}  # synth -> [(ratio, abs_frame_diff)]
    for vc, tgt, syn in pairs:
        try:
            fv, ft = duration_frames(vc), duration_frames(tgt)
        except Exception as e:  # unreadable file -> report once, keep going
            print(f"[duration_check.py] skip duration {vc}: {e}")
            continue
        dur.setdefault(syn, []).append((fv / ft, abs(fv - ft)))

    # 2) Alignment (mel cross-correlation) on the first --xcorr pairs of each synthesizer.
    lag: dict[str, list[tuple[int, float]]] = {}  # synth -> [(best_lag, corr)]
    n_xcorr: dict[str, int] = {}
    for vc, tgt, syn in pairs:
        if n_xcorr.get(syn, 0) >= args.xcorr:
            continue
        n_xcorr[syn] = n_xcorr.get(syn, 0) + 1
        try:
            l, c = best_lag(contour(vc), contour(tgt), args.max_lag)
        except Exception as e:
            print(f"[duration_check.py] skip xcorr {vc}: {e}")
            continue
        lag.setdefault(syn, []).append((l, c))

    # Report.
    hop_ms = HOP_LENGTH / SAMPLE_RATE * 1000
    print(f"Frame = {hop_ms:.1f} ms (hop {HOP_LENGTH} @ {SAMPLE_RATE} Hz). "
          f"Aligned synthesizer => ratio~=1.0, |dframes| small, median |lag|~=0, high corr.\n")
    header = f"{'synth':<10} {'n':>6} {'dur_ratio':>10} {'med|df|':>8} | {'nX':>5} {'medlag':>7} {'med|lag|':>8} {'|lag|<=1':>8} {'medcorr':>8}"
    print(header)
    print("-" * len(header))
    for syn in sorted(dur):
        ratios = [r for r, _ in dur[syn]]
        dframes = [df for _, df in dur[syn]]
        ls = lag.get(syn, [])
        lags = [l for l, _ in ls]
        corrs = [c for _, c in ls]
        near = (sum(1 for l in lags if abs(l) <= 1) / len(lags)) if lags else float("nan")
        print(f"{syn:<10} {len(dur[syn]):>6} {_med(ratios):>10.3f} {_med(dframes):>8.1f} | "
              f"{len(ls):>5} {_med([float(x) for x in lags]):>7.1f} "
              f"{_med([float(abs(x)) for x in lags]):>8.1f} {near:>8.2f} {_med(corrs):>8.3f}")


if __name__ == "__main__":
    main()
