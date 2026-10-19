"""How the quality of the VC input affects the reconstruction, swept over thresholds.

For three properties of the VC sample (its WER, its duration, its similarity to the target
voice) and two reconstruction metrics (EER, WER): keep only the test utterances whose VC
property passes a threshold, recompute the metric on that subset, and repeat over a grid of
thresholds. Duration and target similarity keep `F > t`, VC WER keeps `F <= t`, so a stricter
threshold always removes the low-quality inputs first. The figure has one row per property and
three columns (EER, % of utterances kept, WER), with one curve per (architecture, synthesizer).
The threshold grid of a property is shared by all curves and spans its p1..p99 range.

Inputs, joined on (synthesizer, VC wav file name), since the same file name occurs in every
Test-N folder:
* analysis/results/vc_quality_Test-N.tsv: VC transcripts and target similarity (vc_quality.py).
  The VC duration is read from the corpus and cached in `duration_cache.tsv`.
* outputs/<arch>/synth-N[-align]/test-final_trials_*.tsv: the reconstruction trials of each
  run's final test eval. Only runs trained with `--wer` appear in the WER column.

Both WERs are computed the same way (`capped_wer`). Only Whisper's text normalizer is used, so
this needs the `whisper` package but no GPU.

    python -m analysis.correlate_quality [--archs freevc diffvc] [--synths 1 4]
        [--num-thresholds 10] [--results-dir analysis/results] [--outputs-dir outputs]
        [--slt-root ../datasets/slt-sstc] [--duration-cache analysis/results/duration_cache.tsv]
        [--out analysis/figures/threshold_impact.svg]
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from functools import lru_cache

import numpy as np

from evaluation.asr import english_normalizer, word_error_rate

RUN_RE = re.compile(r"synth-(\d+)(-align)?$")  # run directory name: synth-<N> or synth-<N>-align

# (key, label, direction) of each VC property. "<=" keeps F <= t (VC WER, lower is better),
# ">" keeps F > t (higher is better)
FACTORS = [("wer_vc", "VC WER", "<="), ("target_sim_vc", "VC target sim.", ">"),
           ("duration_vc", "VC duration (s)", ">")]
# (key, label) of each figure column
METRICS = [("eer_recon", "Reconstruction EER"), ("kept_pct", "Utterances kept (%)"),
           ("wer_recon", "Reconstruction WER")]
ARCH_LABELS = {"freevc": "FreeVC", "againvc": "AGAIN-VC", "diffvc": "DiffVC"}  # legend names
_CMAP_NAME = "tab20"  # 20 distinct colors before they repeat

# an EER/WER point is only plotted if at least this many trials pass the threshold, since
# estimates from fewer trials are noisy (the % kept column shows every point)
MIN_TRIALS = 200

_norm = None


@lru_cache(maxsize=1 << 19)
def _normalize(text: str) -> str:
    """Whisper's `EnglishTextNormalizer`, memoized: every curve rescores the same references."""
    global _norm
    if _norm is None:  # built on first use, so importing this module doesn't need whisper
        _norm = english_normalizer()
    return _norm(text)


def capped_wer(ref: str, hyp: str) -> tuple[int, int]:
    """`(edits, n_ref_words)` of one utterance, with the edits capped at `n_ref_words`.

    The single WER definition of this file, used for both the VC WER (x axis) and the
    reconstruction WER (y axis). Text is normalized with Whisper's `EnglishTextNormalizer`, the
    normalizer all stored WERs in the repo were computed with (it keeps numbers, which the
    LibriSpeech references spell out).

    The cap limits a single utterance's WER to 1.0. Only insertions push WER above 1.0, e.g. a
    long ASR hallucination on a short reference, and such an utterance is already entirely wrong.
    About 4% of the VC inputs are above 1.0 and the uncapped p99 is 1.5, so without the cap a
    third of the threshold grid would cover that sparse tail. The capped utterances are short
    (about half the reference words of an average one), so the micro-averaged WER of a run drops
    by at most 1.6 pp (VC) and 2.7 pp (reconstruction). The WERs logged during training stay
    uncapped (plot_main_results.py).
    """
    e, nref = word_error_rate(ref, hyp, normalizer=_normalize)
    return min(e, nref), nref


def compute_eer(target_scores, nontarget_scores) -> float:
    """Equal error rate of target / non-target cosine scores.

    Same as evaluation.asv.compute_eer, copied so this module doesn't import torch.
    """
    target = np.asarray(target_scores, dtype=np.float64)
    nontarget = np.asarray(nontarget_scores, dtype=np.float64)
    if len(target) == 0 or len(nontarget) == 0:
        return float("nan")
    target.sort()
    nontarget.sort()
    thr = np.sort(np.unique(np.concatenate([target, nontarget])))  # every score is a candidate threshold
    frr = np.searchsorted(target, thr, side="left") / len(target)  # targets below the threshold
    far = (len(nontarget) - np.searchsorted(nontarget, thr, side="left")) / len(nontarget)  # non-targets at or above it
    idx = int(np.argmin(np.abs(far - frr)))  # the threshold where both error rates meet
    return float((far[idx] + frr[idx]) / 2.0)


def read_tsv(path):
    """Rows of a TSV written by this repo, as dicts.

    QUOTE_NONE because the files are plain tab-joined: a transcript starting with `"` would
    otherwise make the reader merge the following lines into one row.
    """
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def load_vc_quality(results_dir: str, synths: list[int] | None) -> dict[tuple[int, str], dict]:
    """(synth, VC wav file name) -> {wer_vc, target_sim_vc}, from vc_quality_Test-N.tsv.

    Keyed by synthesizer too, since the same file name occurs in every Test-N folder.
    """
    out = {}
    paths = sorted(glob.glob(os.path.join(results_dir, "vc_quality_Test-*.tsv")))
    for path in paths:
        n = int(os.path.basename(path)[len("vc_quality_Test-"):-len(".tsv")])  # N from the file name
        if synths and n not in synths:
            continue
        for row in read_tsv(path):
            # the stored edit counts were computed with the same normalizer as capped_wer, so
            # only the cap is applied here
            e, nref = int(row["wer_edits"] or 0), int(row["wer_nref"] or 0)
            out[(n, os.path.basename(row["file_path"]))] = {
                "wer_vc": min(e, nref) / nref if nref else float("nan"),  # no reference words -> no WER
                "target_sim_vc": float(row["target_sim"]),
            }
    return out


def find_trial_files(outputs_dir: str, archs: list[str] | None, synths: list[int] | None,
                     align_only: bool = False, prefer_align: list[int] | None = None):
    """Yield (arch, synth, run, path) of each outputs/<arch>/synth-N[-align]/test-final_trials_*.tsv.

    For synthesizers in `prefer_align` only one run is kept: the -align run if there is one,
    else the plain one. Other synthesizers yield all their runs.
    """
    chosen: dict[tuple[str, int], tuple] = {}  # (arch, synth) -> the run kept for prefer_align
    for arch_dir in sorted(glob.glob(os.path.join(outputs_dir, "*"))):
        arch = os.path.basename(arch_dir)
        if archs and arch not in archs:
            continue
        for run_dir in sorted(glob.glob(os.path.join(arch_dir, "synth-*"))):
            run = os.path.basename(run_dir)
            m = RUN_RE.match(run)
            if not m:  # not a training run directory
                continue
            n = int(m.group(1))
            if synths and n not in synths:
                continue
            if align_only and not run.endswith("-align"):
                continue
            paths = glob.glob(os.path.join(run_dir, "test-final_trials_*.tsv"))
            if not paths:  # run without a final test eval
                continue
            # the most recently written one, file names do not sort by epoch (epoch9 > epoch10)
            path = max(paths, key=os.path.getmtime)
            if prefer_align and n in prefer_align:
                # synth-N-align sorts after synth-N, so it overwrites the plain run
                chosen[(arch, n)] = (arch, n, run, path)
                continue
            yield arch, n, run, path
    yield from chosen.values()  # then the one run kept for each prefer_align synthesizer


def load_records(outputs_dir: str, archs: list[str] | None, synths: list[int] | None,
                  vc_quality: dict, align_only: bool = False,
                  prefer_align: list[int] | None = None) -> list[dict]:
    """One dict per reconstruction trial: arch, synth, curve, basename, pos_score, neg_score,
    wer_vc, target_sim_vc, and wer_edits/wer_nref if the run measured WER."""
    records = []
    for arch, synth, run, path in find_trial_files(outputs_dir, archs, synths, align_only, prefer_align):
        rows = read_tsv(path)
        has_wer = rows and "utt_wer" in rows[0]  # the run was trained with --wer
        # an -align run is a different model, so it gets its own curve
        curve = f"{ARCH_LABELS.get(arch, arch)} VC-{synth}" + (" Aligned" if run.endswith("-align") else "")
        for row in rows:
            basename = os.path.basename(row["recon_source_vc"])  # the VC wav this trial reconstructed
            vc = vc_quality.get((synth, basename))
            if vc is None:  # synthesizer not in vc_quality_Test-N.tsv
                continue
            rec = {"arch": arch, "synth": synth, "curve": curve, "basename": basename, **vc,
                   "pos_score": float(row["pos_score"]), "neg_score": float(row["neg_score"])}
            if has_wer and row.get("ref_text") and row.get("hyp_text"):
                # recomputed from the transcripts, so the cap applies
                rec["wer_edits"], rec["wer_nref"] = capped_wer(row["ref_text"], row["hyp_text"])
            records.append(rec)
    return records


def fill_durations(records: list[dict], slt_root: str, cache_path: str) -> None:
    """Add `duration_vc` (seconds) to every record, through a (synth, file name) -> seconds cache.

    Keyed by synthesizer too, since each Test-N folder holds a different VC output under the
    same file name.
    """
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for row in csv.reader(f, delimiter="\t"):  # synth, file name, seconds
                cache[(int(row[0]), row[1])] = float(row[2])

    needed = {(r["synth"], r["basename"]) for r in records if (r["synth"], r["basename"]) not in cache}
    if needed:
        import librosa  # only needed for utterances not cached yet

        print(f"[correlate_quality.py] measuring the duration of {len(needed)} new utterance(s)")
        for synth, basename in sorted(needed):
            path = os.path.join(slt_root, "test", f"Test-{synth}", basename)
            cache[(synth, basename)] = float(librosa.get_duration(path=path))  # from the file header
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "w") as f:  # rewrite the whole cache, old and new entries
            for (synth, basename), seconds in sorted(cache.items()):
                f.write(f"{synth}\t{basename}\t{seconds}\n")

    for r in records:
        r["duration_vc"] = cache[(r["synth"], r["basename"])]


def thresholds_for(records: list[dict], factor_key: str, num_thresholds: int,
                   direction: str) -> np.ndarray:
    """`num_thresholds` evenly spaced thresholds over the p1..p99 range of a property, ascending.

    The percentiles keep a few extreme values from stretching the grid over a range with
    almost no data. The loosest threshold keeps about all utterances: the lowest one for ">",
    the highest one for "<=".
    """
    vals = np.array([r[factor_key] for r in records if not np.isnan(r[factor_key])], dtype=float)
    if vals.size == 0:
        return np.array([])
    lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
    if direction == ">":
        return np.linspace(lo, hi, num_thresholds, endpoint=False)  # lo .. just below hi
    return np.linspace(hi, lo, num_thresholds, endpoint=False)[::-1]  # just above lo .. hi


def _passing(records: list[dict], factor_key: str, t: float, direction: str) -> list[dict]:
    """Records whose property passes threshold `t`. A missing (NaN) value never passes."""
    if direction == ">":
        return [r for r in records if not np.isnan(r[factor_key]) and r[factor_key] > t]
    # "<=" is inclusive: the cap puts every completely wrong VC transcript at exactly 1.0, and
    # a strict "<" would drop all of them even at the loosest threshold
    return [r for r in records if not np.isnan(r[factor_key]) and r[factor_key] <= t]


def eer_curve(records: list[dict], factor_key: str, thresholds: np.ndarray,
              direction: str, min_trials: int = MIN_TRIALS) -> np.ndarray:
    """Reconstruction EER of the passing trials at each threshold (NaN below `min_trials`)."""
    out = np.full(len(thresholds), np.nan)
    for i, t in enumerate(thresholds):
        subset = _passing(records, factor_key, t, direction)
        if len(subset) < min_trials:
            continue
        # each trial contributes one target score (vs its source speaker) and one non-target score
        out[i] = compute_eer([r["pos_score"] for r in subset], [r["neg_score"] for r in subset])
    return out


def wer_curve(records: list[dict], factor_key: str, thresholds: np.ndarray,
              direction: str, min_trials: int = MIN_TRIALS) -> np.ndarray:
    """Reconstruction WER of the passing trials at each threshold (NaN below `min_trials`)."""
    wer_records = [r for r in records if "wer_edits" in r]  # only trials with a transcript
    out = np.full(len(thresholds), np.nan)
    for i, t in enumerate(thresholds):
        subset = _passing(wer_records, factor_key, t, direction)
        if len(subset) < min_trials:
            continue
        nref = sum(r["wer_nref"] for r in subset)
        if nref == 0:
            continue
        out[i] = sum(r["wer_edits"] for r in subset) / nref  # total edits / total reference words
    return out


def retention_curve(records: list[dict], factor_key: str, thresholds: np.ndarray,
                    direction: str) -> np.ndarray:
    """% of the curve's trials that pass each threshold.

    The denominator is all trials, so those with a missing property value count as dropped.
    """
    if not records:
        return np.full(len(thresholds), np.nan)
    return np.array([100.0 * len(_passing(records, factor_key, t, direction)) / len(records)
                     for t in thresholds])


def plot(records: list[dict], num_thresholds: int, out_path: str):
    """Draw the FACTORS x METRICS grid of threshold curves to `out_path`."""
    import matplotlib

    matplotlib.use("Agg")  # no display on the cluster
    import matplotlib.pyplot as plt

    curves = sorted({r["curve"] for r in records})
    wer_curves = sorted({r["curve"] for r in records if "wer_edits" in r})  # runs trained with --wer
    cmap = plt.get_cmap(_CMAP_NAME)
    colors = {c: cmap(i % cmap.N) for i, c in enumerate(curves)}  # one fixed color per curve in every panel

    curve_fns = {"eer_recon": eer_curve, "wer_recon": wer_curve, "kept_pct": retention_curve}
    fig, axes = plt.subplots(len(FACTORS), len(METRICS), figsize=(13, 10))
    for i, (factor_key, factor_label, direction) in enumerate(FACTORS):
        thresholds = thresholds_for(records, factor_key, num_thresholds, direction)  # shared by the row
        for j, (metric_key, metric_label) in enumerate(METRICS):
            ax = axes[i, j]
            curve_list = wer_curves if metric_key == "wer_recon" else curves
            for curve in curve_list:
                by_curve = [r for r in records if r["curve"] == curve]
                y = curve_fns[metric_key](by_curve, factor_key, thresholds, direction)
                ax.plot(thresholds, y, color=colors[curve], label=curve, marker="o", ms=2.5, lw=1.2)
            if j == 1:  # the three columns share the x axis, so only the middle one is labeled
                ax.set_xlabel(f"{factor_label} threshold t (keep {direction} t)", fontsize=13)
            ax.set_ylabel(metric_label, fontsize=13)
            ax.set_ylim(bottom=0)  # all three metrics are non-negative
            ax.grid(color="0.85", lw=0.5, zorder=0)
            ax.tick_params(labelsize=12)
            # for "<=" a larger t keeps more data, so flip the axis: in every row the loosest
            # threshold is on the left and the thresholds get stricter to the right
            if direction != ">":
                ax.invert_xaxis()

    handles, labels = axes[0, 0].get_legend_handles_labels()  # the EER panel has every curve
    ncol = min(len(labels), 6) or 1
    fig.legend(handles, labels, loc="upper center", ncol=ncol, frameon=False,
               fontsize=12, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))  # leave the top for the legend
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    print(f"[correlate_quality.py] wrote {out_path}")


def self_test():
    """Checks of the threshold, metric and WER logic on hand-made records."""
    recs = [
        {"duration_vc": 1.0, "pos_score": 0.3, "neg_score": 0.7},  # a badly reconstructed short utterance
        {"duration_vc": 2.0, "pos_score": 0.9, "neg_score": 0.1},
        {"duration_vc": 3.0, "pos_score": 0.8, "neg_score": 0.2},
        {"duration_vc": 4.0, "pos_score": 0.85, "neg_score": 0.15},
    ]
    thresholds = np.array([0.5, 1.5, 3.5])
    got = eer_curve(recs, "duration_vc", thresholds, ">", min_trials=2)
    # t=0.5 keeps all four (EER > 0), t=1.5 drops the bad one (EER 0), t=3.5 keeps one, below min_trials
    assert got[1] == 0.0 and got[0] > got[1] and np.isnan(got[2]), got
    expected_all = compute_eer([0.3, 0.9, 0.8, 0.85], [0.7, 0.1, 0.2, 0.15])
    assert abs(got[0] - expected_all) < 1e-9, (got[0], expected_all)

    # "<=": the high-WER utterance is the bad one, t=0.9 keeps it (inclusive) and t=0.5 drops it
    lt_recs = [
        {"wer_vc": 0.1, "pos_score": 0.9, "neg_score": 0.1},
        {"wer_vc": 0.2, "pos_score": 0.8, "neg_score": 0.2},
        {"wer_vc": 0.9, "pos_score": 0.3, "neg_score": 0.7},
    ]
    lt = eer_curve(lt_recs, "wer_vc", np.array([0.5, 0.9]), "<=", min_trials=1)
    assert lt[0] == 0.0 and lt[1] > lt[0], lt
    assert len(_passing(lt_recs, "wer_vc", 0.9, "<=")) == 3

    # WER is total edits over total reference words: (1 + 5) / 20, then 5 / 10
    wer_recs = [
        {"duration_vc": 1.0, "wer_edits": 1, "wer_nref": 10},
        {"duration_vc": 2.0, "wer_edits": 5, "wer_nref": 10},
    ]
    w = wer_curve(wer_recs, "duration_vc", np.array([0.5, 1.5]), ">", min_trials=1)
    assert abs(w[0] - 6 / 20) < 1e-9 and abs(w[1] - 5 / 10) < 1e-9, w

    # normalization ignores case and punctuation and matches numerals to spelled-out numbers;
    # a 200-word hypothesis on a 3-word reference counts 3 edits
    assert capped_wer("A B C", "a, b! d") == (1, 3), capped_wer("A B C", "a, b! d")
    assert capped_wer("SIX O'CLOCK", "6 o'clock") == (0, 2), capped_wer("SIX O'CLOCK", "6 o'clock")
    assert capped_wer("A B C", " ".join(["x"] * 200)) == (3, 3)
    assert capped_wer("", "anything") == (0, 0)  # no reference, the caller turns it into NaN

    # % kept: t=0.5 keeps 4/4, t=1.5 keeps 3/4, t=3.5 keeps 1/4
    kept = retention_curve(recs, "duration_vc", thresholds, ">")
    assert list(kept) == [100.0, 75.0, 25.0], kept

    # the grid is ascending, ignores NaN and stays inside the data range, in both directions
    dur_recs = [{"duration_vc": 1.0}, {"duration_vc": 3.0}, {"duration_vc": float("nan")}]
    for d in (">", "<="):
        t = thresholds_for(dur_recs, "duration_vc", 4, d)
        assert len(t) == 4 and 1.0 <= t[0] < t[-1] <= 3.0 and np.all(np.diff(t) > 0), t
    # one extreme value (<1% of the data) doesn't stretch the grid
    wer_recs = [{"wer_vc": v} for v in list(np.linspace(0.0, 1.0, 200)) + [68.0]]
    t = thresholds_for(wer_recs, "wer_vc", 4, "<=")
    assert t.max() < 2.0, t

    print("[correlate_quality.py] self-test OK")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default="analysis/results")
    p.add_argument("--outputs-dir", default="outputs")
    p.add_argument("--slt-root", default="../datasets/slt-sstc")
    p.add_argument("--duration-cache", default=None,
                   help="(synth, file name) -> seconds cache tsv (default <results-dir>/duration_cache.tsv)")
    p.add_argument("--archs", nargs="+", default=["freevc"],
                   help="reconstructor architectures to include (default: freevc)")
    p.add_argument("--synths", nargs="+", type=int, default=None,
                   help="test synthesizer numbers to include, e.g. --synths 1 4 (default: all present)")
    p.add_argument("--align-only", action="store_true",
                   help="keep only the -align runs (drop the plain synth-N curves)")
    p.add_argument("--prefer-align", nargs="+", type=int, default=[1, 4], metavar="N",
                   help="synths reduced to one curve, the -align run where present, else the plain "
                        "one (default: 1 4). Other synths keep all their runs")
    p.add_argument("--num-thresholds", type=int, default=10,
                   help="thresholds per property, evenly spaced over its p1..p99 range")
    p.add_argument("--out", default="analysis/figures/threshold_impact.svg")
    p.add_argument("--self-test", action="store_true", help="run the self-test and exit")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return

    vc_quality = load_vc_quality(args.results_dir, args.synths)
    if not vc_quality:
        raise SystemExit(f"[correlate_quality.py] no vc_quality_Test-*.tsv in {args.results_dir}, "
                         "run vc_quality.py first")

    records = load_records(args.outputs_dir, args.archs, args.synths, vc_quality,
                           args.align_only, args.prefer_align)
    if not records:
        raise SystemExit("[correlate_quality.py] no reconstruction trials found, run the final test "
                         "eval of at least one train_<arch>.py first")

    fill_durations(records, args.slt_root, args.duration_cache or os.path.join(args.results_dir, "duration_cache.tsv"))

    n_curves = len({r["curve"] for r in records})
    n_wer_curves = len({r["curve"] for r in records if "wer_edits" in r})
    print(f"[correlate_quality.py] {len(records)} trials in {n_curves} (arch, synth) curves "
          f"({n_wer_curves} with reconstruction WER)")

    plot(records, args.num_thresholds, args.out)


if __name__ == "__main__":
    main()
