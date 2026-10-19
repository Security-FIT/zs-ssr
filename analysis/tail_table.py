"""Rows of the paper's tail table: the VC inputs behind the best, median and worst reconstructions.

Each FreeVC test run (outputs/freevc/synth-N[-align]/test-final_trials_*.tsv, the -align run where
there is one) is ranked by the positive-trial score, the cosine between a reconstruction and a
real utterance of its source speaker. The top, middle and bottom `--frac` of every run form the
Best, Median and Worst groups. Ranking per run keeps one hard synthesizer from filling a group.
The groups are then pooled over runs and their VC inputs described by:

* mean duration (s), from analysis/results/duration_cache.tsv
* micro WER (%) of the VC input against the source transcript, from vc_quality_Test-N.tsv
* EER (%) of the VC input against its target VoxCeleb speaker, from the per-utterance
  target_sim / nontarget_sim of vc_quality_Test-N.tsv

Each group prints as a LaTeX table row.

    python -m analysis.tail_table [--frac 0.10] [--outputs-dir outputs]
        [--results-dir analysis/results] [--synths 1 ... 8]
"""

from __future__ import annotations

import argparse
import csv
import os

from analysis.correlate_quality import compute_eer, find_trial_files, read_tsv

GROUPS = ("Best", "Median", "Worst")


def group_indices(order_desc: list[int], k: int) -> dict[str, list[int]]:
    """Best, around-median and worst `k` indices of an order sorted by descending score."""
    lo = max(0, (len(order_desc) - k) // 2)  # start of the k-wide window centred on the median
    return {"Best": order_desc[:k], "Median": order_desc[lo:lo + k], "Worst": order_desc[-k:]}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs-dir", default="outputs")
    p.add_argument("--results-dir", default="analysis/results")
    p.add_argument("--synths", nargs="+", type=int, default=list(range(1, 9)))
    p.add_argument("--frac", type=float, default=0.10,
                   help="fraction of each run's ranked trials per group")
    args = p.parse_args()

    # one trials TSV per synthesizer, the -align run where there is one
    run_paths = {n: (run, path) for _, n, run, path in
                 find_trial_files(args.outputs_dir, ["freevc"], args.synths, prefer_align=args.synths)}
    if not run_paths:
        raise SystemExit(f"[tail_table.py] no freevc test-final_trials_*.tsv under {args.outputs_dir}")

    # VC durations measured by correlate_quality.py, rows are (synth, file name, seconds)
    durations = {}
    with open(os.path.join(args.results_dir, "duration_cache.tsv")) as f:
        for row in csv.reader(f, delimiter="\t"):
            durations[(int(row[0]), row[1])] = float(row[2])

    # per group: durations, summed WER edits and reference words, target and nontarget scores
    acc = {g: {"dur": [], "edits": 0, "nref": 0, "tgt": [], "non": []} for g in GROUPS}
    for n, (run, trials_path) in sorted(run_paths.items()):
        vc_wer = {}    # VC file name -> (edits, reference words) of its transcript
        vc_trial = {}  # VC file name -> (target, nontarget) cosine to VoxCeleb clips
        for r in read_tsv(os.path.join(args.results_dir, f"vc_quality_Test-{n}.tsv")):
            base = os.path.basename(r["file_path"])
            if r["wer_edits"] not in ("", "nan"):  # empty when there is no reference transcript
                vc_wer[base] = (int(r["wer_edits"]), int(r["wer_nref"]))
            vc_trial[base] = (float(r["target_sim"]), float(r["nontarget_sim"]))

        # rank this run's trials by the reconstruction's positive score, best first
        rows = read_tsv(trials_path)
        order = sorted(range(len(rows)), key=lambda j: float(rows[j]["pos_score"]), reverse=True)
        k = max(1, int(len(rows) * args.frac))
        print(f"[tail_table.py] Test-{n} ({run}): {len(rows)} trials, {k} per group")
        for g, idxs in group_indices(order, k).items():
            a = acc[g]
            for j in idxs:
                # the VC input this reconstruction was made from
                base = os.path.basename(rows[j]["recon_source_vc"])
                if (n, base) in durations:  # the cache holds only utterances joined in correlate_quality.py
                    a["dur"].append(durations[(n, base)])
                if base in vc_wer:
                    a["edits"] += vc_wer[base][0]
                    a["nref"] += vc_wer[base][1]
                if base in vc_trial:
                    a["tgt"].append(vc_trial[base][0])
                    a["non"].append(vc_trial[base][1])

    print(f"\n[tail_table.py] pooled over runs: {', '.join(r for r, _ in sorted(run_paths.values()))}")
    for g in GROUPS:
        a = acc[g]
        dur = sum(a["dur"]) / len(a["dur"])
        wer = 100.0 * a["edits"] / a["nref"]  # micro WER over the whole group
        eer = 100.0 * compute_eer(a["tgt"], a["non"])
        # n = utterances with a known duration
        print(f"{g:6s} & {dur:.1f} & {wer:.1f} & {eer:.1f} \\\\   (n={len(a['dur'])})")


if __name__ == "__main__":
    main()
