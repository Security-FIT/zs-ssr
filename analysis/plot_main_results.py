"""Per-system result figures for the paper: grouped bars over Test-1..8, SVGs in analysis/figures/.

* per_system_eer.svg: target-speaker EER of the VC output, FreeVC reconstruction EER (plain and
  aligned run) and the baseline source-speaker verifier (SSV) EER.
* per_system_wer.svg: WER of the VC output next to the FreeVC reconstruction WER.
* per_system_mos.svg: mean UTMOS22 predicted MOS of the VC output and of the reconstruction.

Sources of each series:
* VC EER/WER: analysis/results/vc_quality_Test-N.tsv (`vc_series`).
* FreeVC EER/WER: the last `test-final:` line of outputs/freevc/synth-N[-align]/val_metrics.txt
  (`freevc_series`).
* MOS: the naturalness TSVs in analysis/results/ (`mos_series`).
* SSV EER: quoted from the SSTC 2024 challenge paper, Table 3, "Multi-VC-8 + Adapter".

    python -m analysis.plot_main_results
"""

from __future__ import annotations

import glob
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.correlate_quality import compute_eer, read_tsv

SYSTEMS = [1, 2, 3, 4, 5, 6, 7, 8]
OUT_DIR = "analysis/figures"

# EER in percent for Test-1..8, the only series not computed from files in this repo
SSV_EER = [9.950, 10.402, 6.943, 7.636, 6.683, 10.783, 32.392, 28.418]

# Okabe-Ito colours (colourblind safe): VC bluish green, SSV blue, FreeVC orange, aligned vermillion
C_VC, C_SSV, C_EER, C_ALIGN = "#009E73", "#0072B2", "#E69F00", "#D55E00"

# hatches tell the series apart in grayscale print: VC backslashes, SSV solid, FreeVC diagonals,
# aligned dots
H_VC, H_SSV, H_EER, H_ALIGN = "\\\\\\", "", "///", "..."

RC = {
    "font.size": 8, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "figure.dpi": 200, "hatch.linewidth": 0.5,
}


def vc_series(results_dir: str = "analysis/results"):
    """(eer, wer) of the VC output in percent, one value per system, from vc_quality_Test-N.tsv.

    EER is over the per-utterance target_sim / nontarget_sim trial pairs. WER is micro averaged
    (total edits / total reference words). Both match the `[vc_quality.py]` summary of each job.
    """
    eer, wer = [], []
    for s in SYSTEMS:
        rows = read_tsv(os.path.join(results_dir, f"vc_quality_Test-{s}.tsv"))
        # each row is one trial pair: (VC output, target clip) and (VC output, another voice's clip)
        eer.append(100 * compute_eer([float(r["target_sim"]) for r in rows],
                                     [float(r["nontarget_sim"]) for r in rows]))
        # rows without a reference transcript have an empty wer_nref and do not count
        q = [r for r in rows if r["wer_nref"] not in ("", "nan")]
        wer.append(100 * sum(int(r["wer_edits"]) for r in q) / sum(int(r["wer_nref"]) for r in q))
    return eer, wer


def freevc_series(outputs_dir: str = "outputs"):
    """(eer, wer, align_eer, align_wer) of FreeVC in percent, None for a run that is missing.

    Read from the last `test-final:` line of outputs/freevc/synth-N[-align]/val_metrics.txt, so a
    re-evaluated run shows up in the next plot.
    """
    out = [[None] * len(SYSTEMS) for _ in range(4)]
    for i, s in enumerate(SYSTEMS):
        # the plain run fills the first two lists, the -align run the last two
        for (E, W), run in ((out[:2], f"synth-{s}"), (out[2:], f"synth-{s}-align")):
            path = os.path.join(outputs_dir, "freevc", run, "val_metrics.txt")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                lines = [ln for ln in f if "test-final:" in ln]
            if not lines:  # the run never reached its final test evaluation
                continue
            # val_metrics.txt stores fractions, e.g. "EER=0.1234"
            for arr, key in ((E, "EER"), (W, "WER")):
                m = re.search(rf"{key}=([\d.]+)", lines[-1])
                if m:
                    arr[i] = 100 * float(m.group(1))
    return out


def mos_series(results_dir: str = "analysis/results"):
    """(vc, recon, align) mean UTMOS22 per system, None where that system was not scored.

    VC output: naturalness_table.tsv (all systems in one file, the vc_naturalness.py default)
    and/or the per-system naturalness_Test-N.tsv shards of the SGE jobs, a shard wins over the
    table. Reconstruction: recon_naturalness_freevc_<run>.tsv, one file per run.
    """
    vc = {}  # synth -> naturalness row
    table = os.path.join(results_dir, "naturalness_table.tsv")
    if os.path.exists(table):
        for r in read_tsv(table):
            vc[int(r["synth"])] = r
    for path in sorted(glob.glob(os.path.join(results_dir, "naturalness_Test-*.tsv"))):
        r = read_tsv(path)[0]  # a shard holds a single row
        vc[int(r["synth"])] = r

    recon, align = {}, {}
    for path in sorted(glob.glob(os.path.join(results_dir, "recon_naturalness_freevc_*.tsv"))):
        r = read_tsv(path)[0]
        (align if r["run"].endswith("-align") else recon)[int(r["synth"])] = r

    def mean_by_system(rows: dict):
        return [float(rows[s]["utmos22_mean"]) if s in rows else None for s in SYSTEMS]

    return mean_by_system(vc), mean_by_system(recon), mean_by_system(align)


def plot(series, ylabel: str, out_name: str, fmt: str = ".1f", headroom: float = 1.18,
         xlabel: str | None = None, legend_kw: dict | None = None):
    """Grouped bars, one cluster per system, written to OUT_DIR/out_name.

    `series` is a list of (label, values, colour, hatch) with one value per system. A None value
    leaves no bar, and the bars present in a cluster are re-centred on its tick. Each bar is
    labelled with its value in `fmt`. The y axis reaches `headroom` times the tallest bar to leave
    room for the legend.
    """
    plt.rcParams.update(RC)
    vals = np.array([[np.nan if v is None else float(v) for v in s[1]] for s in series])
    n = len(series)
    x = np.arange(len(SYSTEMS), dtype=float)  # cluster centres
    bw = 0.86 / n                             # cluster width split over the series

    # place only the bars present in each cluster, side by side around its centre
    xpos = np.tile(x, (n, 1))
    for i in range(len(SYSTEMS)):
        present = [k for k in range(n) if np.isfinite(vals[k, i])]
        for j, k in enumerate(present):
            xpos[k, i] = x[i] + (j - (len(present) - 1) / 2) * bw

    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    for k, (label, _, color, hatch) in enumerate(series):
        bars = ax.bar(xpos[k], vals[k], width=bw * 0.9, color=color, label=label,
                      hatch=hatch, edgecolor="white", linewidth=0.4, zorder=3)
        # NaN bars have zero height, so give them an empty label
        ax.bar_label(bars, labels=[format(v, fmt) if np.isfinite(v) else "" for v in vals[k]],
                     fontsize=5, padding=1, color="0.25")

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(s) for s in SYSTEMS])
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylim(0, np.nanmax(vals) * headroom)
    ax.margins(x=0.01)
    ax.grid(axis="y", color="0.85", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    # large legend swatches so the hatch pattern is recognisable
    ax.legend(frameon=False, ncol=n, fontsize=7, loc="upper left", borderaxespad=0.2,
              **(legend_kw or dict(handlelength=2.4, handleheight=1.6, columnspacing=1.1)))

    os.makedirs(OUT_DIR, exist_ok=True)
    svg = os.path.join(OUT_DIR, out_name)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[plot_main_results.py] wrote {svg}")


def main():
    vc_eer, vc_wer = vc_series()
    freevc_eer, freevc_wer, align_eer, align_wer = freevc_series()
    vc_mos, recon_mos, align_mos = mos_series()

    # the EER and WER figures sit above the MOS figure in the paper, which carries the shared
    # x axis title
    plot([("VC target EER", vc_eer, C_VC, H_VC),
          ("FreeVC SSR EER", freevc_eer, C_EER, H_EER),
          ("FreeVC Aligned EER", align_eer, C_ALIGN, H_ALIGN),
          ("SSV EER", SSV_EER, C_SSV, H_SSV)],
         "EER (%)", "per_system_eer.svg")
    plot([("VC input WER", vc_wer, C_VC, H_VC),
          ("FreeVC SSR WER", freevc_wer, C_EER, H_EER),
          ("FreeVC SSR Aligned WER", align_wer, C_ALIGN, H_ALIGN)],
         "WER (%)", "per_system_wer.svg")
    if any(v is not None for v in vc_mos):
        plot([("VC input", vc_mos, C_VC, H_VC),
              ("FreeVC SSR", recon_mos, C_EER, H_EER),
              ("FreeVC SSR Aligned", align_mos, C_ALIGN, H_ALIGN)],
             "Predicted MOS (UTMOS22)", "per_system_mos.svg", fmt=".2f", headroom=1.25,
             xlabel="VC synthesizer",
             legend_kw=dict(handlelength=1.8, handleheight=1.4, columnspacing=0.9))
    else:
        print("[plot_main_results.py] skip the MOS figure: no VC naturalness TSVs "
              "(run scripts/sge/analysis/vc_naturalness.sh)")

    def mean(vals):
        present = [v for v in vals if v is not None]  # a missing run counts as no value
        return np.mean(present) if present else float("nan")

    print(f"[plot_main_results.py] VC EER mean={mean(vc_eer):.2f}%  SSV mean={mean(SSV_EER):.2f}%  "
          f"FreeVC EER mean={mean(freevc_eer):.2f}%  "
          f"VC WER mean={mean(vc_wer):.2f}%  FreeVC WER mean={mean(freevc_wer):.2f}%")


if __name__ == "__main__":
    main()
