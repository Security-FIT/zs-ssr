"""Plot of the in-loop validation metrics, redrawn after every validation.

Each `train_<model>.py` appends one line per validation to `<out_dir>/val_metrics.txt`
(EER from `evaluation/reconstruction_eval.py`), e.g.

    epoch 1 samples 80000 dev-epoch1: EER=0.2345 (n=200 mean_tgt=0.512 mean_non=0.201)

This module parses that log and overwrites `<out_dir>/val_eer.svg`, so the progress of a
running job can be followed from storage. The x axis is the epoch, the unit the checkpoints
are named with (`<model>_ssr_epoch<E>.pth`). Lines in the `step N samples M tag: ...` format
are parsed too, with the epoch taken from their `dev-epochN` tag, so a log that mixes both
formats plots on one epoch axis.

The plot depends only on `val_metrics.txt`, so it survives a resume and can be redrawn at any
time:

    python -m evaluation.plot_metrics <out_dir>     # redraw val_eer.svg from val_metrics.txt
    python -m evaluation.plot_metrics               # self-test on a synthetic log

matplotlib is imported lazily. Without it `render_eer_plot` prints a hint and returns None, so
plotting never stops a training run.
"""

from __future__ import annotations

import math
import os
import re

# epoch {E} samples {N} {tag}: EER={eer} (n={n} mean_tgt={mt} mean_non={mn}) [WER={wer}]
# step {S} [samples {N}] {tag}: EER=...
# The leading counters are all optional, so one regex reads both formats. A line needs at
# least one of them to become a record (checked in parse_val_metrics).
_NUM = r"(nan|[-+0-9.eE]+)"  # a float or nan
_LINE = re.compile(
    rf"(?:epoch\s+(\d+)\s+)?(?:step\s+(\d+)\s+)?(?:samples\s+(\d+)\s+)?(\S+):\s+EER={_NUM}"
    rf"\s+\(n=(\d+)\s+mean_tgt={_NUM}\s+mean_non={_NUM}\)"
    rf"(?:\s+WER={_NUM})?"
)
_TAG_EPOCH = re.compile(r"-epoch(\d+)$")  # step-format lines carry the epoch only in their tag


def _split_of(tag: str) -> str:
    """Group a tag (dev-epoch3 / dev-final / test-final) into its split series."""
    return "test" if tag.startswith("test") else "dev"


def parse_val_metrics(path: str) -> dict[str, list[dict]]:
    """`val_metrics.txt` -> {split: [records sorted by epoch, then samples]}.

    A record is {epoch, step, samples, tag, eer, mean_tgt, mean_non, wer}, with None for the
    fields a line does not have. A step-format line takes its epoch from its `dev-epochN` tag,
    or, for `*-final` tags, the last epoch seen before it (the log is append-only). `EER=nan`
    is kept as NaN. Lines that do not match are skipped.
    """
    series: dict[str, list[dict]] = {}
    if not os.path.exists(path):
        return series
    last_epoch: int | None = None
    with open(path) as f:
        for line in f:
            m = _LINE.search(line)
            if not m:
                continue
            epoch, step, samples, tag, eer, _n, mt, mn, wer = m.groups()
            if epoch is None:  # step-format line, take the epoch from its tag or the line before
                tm = _TAG_EPOCH.search(tag)
                epoch = tm.group(1) if tm else last_epoch
            if epoch is None and step is None and samples is None:
                continue  # no usable x position at all
            if epoch is not None:
                last_epoch = int(epoch)
            series.setdefault(_split_of(tag), []).append({
                "epoch": int(epoch) if epoch is not None else None,
                "step": int(step) if step is not None else None,
                "samples": int(samples) if samples is not None else None,
                "tag": tag,
                "eer": float(eer),
                "mean_tgt": float(mt),
                "mean_non": float(mn),
                "wer": float(wer) if wer is not None else None,
            })
    for recs in series.values():  # epoch first, then samples (or step) within an epoch
        recs.sort(key=lambda r: (r["epoch"] if r["epoch"] is not None else 0,
                                 r["samples"] if r["samples"] is not None else (r["step"] or 0)))
    return series


def render_eer_plot(out_dir: str, metrics_name: str = "val_metrics.txt",
                    out_name: str = "val_eer.svg") -> str | None:
    """Draw `<out_dir>/<out_name>` from `<out_dir>/<metrics_name>`, overwriting it.

    Panels: the EER per split, the mean target and non-target cosine (smoother than the EER
    of a small validation subset), and the WER if any line has one.
    Returns the SVG path, or None when there is nothing to plot or no matplotlib.
    """
    series = parse_val_metrics(os.path.join(out_dir, metrics_name))
    if not series:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless compute node: render to file, no display
        import matplotlib.pyplot as plt
    except Exception as e:  # matplotlib not installed in the training env
        print(f"[plot_metrics.py] matplotlib unavailable ({e}), skipping the EER plot. "
              "Install it with 'uv pip install matplotlib'")
        return None

    # x axis: the epoch when every record has one, else samples seen, else the optimizer step
    # (a step-format log with only *-final lines has no epoch)
    recs_all = [r for recs in series.values() for r in recs]
    if all(r["epoch"] is not None for r in recs_all):
        xkey, xlabel = "epoch", "epoch"
    elif all(r["samples"] is not None for r in recs_all):
        xkey, xlabel = "samples", "samples seen  (optimizer steps x batch size)"
    else:
        xkey, xlabel = "step", "optimizer step"

    colors = {"dev": "tab:blue", "test": "tab:red"}
    # a WER panel only when some record has a WER (runs with --wer)
    have_wer = any(r.get("wer") is not None for r in recs_all)
    nrows = 3 if have_wer else 2
    fig, axes = plt.subplots(nrows, 1, sharex=True, figsize=(8, 3.5 * nrows))
    ax_eer, ax_cos = axes[0], axes[1]
    ax_wer = axes[2] if have_wer else None

    for split in sorted(series):
        recs = series[split]
        c = colors.get(split, "tab:green")
        x = [r[xkey] for r in recs]
        # EER, without NaN points so the line stays connected
        ex = [r[xkey] for r in recs if not math.isnan(r["eer"])]
        ey = [r["eer"] for r in recs if not math.isnan(r["eer"])]
        if ex:
            ax_eer.plot(ex, ey, "-o", color=c, label=split)
            ax_eer.annotate(f"{ey[-1]:.3f}", (ex[-1], ey[-1]), color=c, fontsize=8,
                            textcoords="offset points", xytext=(4, 4))
        # score separation: target (solid) vs non-target (dashed) cosine
        ax_cos.plot(x, [r["mean_tgt"] for r in recs], "-o", color=c, label=f"{split} target")
        ax_cos.plot(x, [r["mean_non"] for r in recs], "--s", color=c, label=f"{split} non-target")
        # WER, only the records that have one
        if ax_wer is not None:
            wx = [r[xkey] for r in recs if r.get("wer") is not None and not math.isnan(r["wer"])]
            wy = [r["wer"] for r in recs if r.get("wer") is not None and not math.isnan(r["wer"])]
            if wx:
                ax_wer.plot(wx, wy, "-o", color=c, label=split)
                ax_wer.annotate(f"{wy[-1]:.3f}", (wx[-1], wy[-1]), color=c, fontsize=8,
                                textcoords="offset points", xytext=(4, 4))

    all_eer = [r["eer"] for recs in series.values() for r in recs if not math.isnan(r["eer"])]
    ax_eer.axhline(0.5, color="grey", ls=":", lw=0.8, label="chance (0.5)")
    ax_eer.set_ylim(0.0, max(0.55, max(all_eer) * 1.15) if all_eer else 0.6)
    ax_eer.set_ylabel("validation EER  (lower = better)")
    ax_eer.set_title("In-loop ASV: reconstruction vs real source speaker")
    ax_eer.grid(True, alpha=0.3)
    ax_eer.legend(loc="best", fontsize=8)

    ax_cos.set_ylabel("mean cosine (recon vs ref)")
    ax_cos.grid(True, alpha=0.3)
    ax_cos.legend(loc="best", fontsize=8)

    if ax_wer is not None:
        all_wer = [r["wer"] for recs in series.values() for r in recs
                   if r.get("wer") is not None and not math.isnan(r["wer"])]
        ax_wer.set_ylim(0.0, max(1.0, max(all_wer) * 1.15) if all_wer else 1.0)
        ax_wer.set_ylabel("WER  (lower = better)")
        ax_wer.grid(True, alpha=0.3)
        ax_wer.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel(xlabel)

    fig.tight_layout()
    out_path = os.path.join(out_dir, out_name)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def main():
    """Redraw the plot of an output directory, or self-test on a synthetic log without arguments."""
    import sys
    import tempfile

    if len(sys.argv) > 1:  # redraw from a real run's output dir
        out = render_eer_plot(sys.argv[1])
        print(f"[plot_metrics.py] wrote {out}" if out else
              "[plot_metrics.py] nothing to plot (no val_metrics.txt or no matplotlib)")
        return

    # a log mixing step-format and epoch-format lines parses onto one epoch axis
    d = tempfile.mkdtemp(prefix="eerplot_")
    lines = [
        "step 1000 samples 16000 dev-epoch1: EER=0.4123 (n=200 mean_tgt=0.310 mean_non=0.220) WER=0.812",
        "step 2000 samples 32000 dev-epoch2: EER=0.3310 (n=200 mean_tgt=0.402 mean_non=0.190) WER=0.641",
        "epoch 3 samples 48000 dev-epoch3: EER=0.2715 (n=200 mean_tgt=0.481 mean_non=0.171) WER=0.523",
        "epoch 3 samples 48000 dev-final:  EER=0.2680 (n=500 mean_tgt=0.490 mean_non=0.168) WER=0.501",
        "epoch 3 samples 48000 test-final: EER=0.2901 (n=500 mean_tgt=0.450 mean_non=0.181) WER=0.560",
    ]
    with open(os.path.join(d, "val_metrics.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    series = parse_val_metrics(os.path.join(d, "val_metrics.txt"))
    assert [r["epoch"] for r in series["dev"]] == [1, 2, 3, 3], series
    assert [r["epoch"] for r in series["test"]] == [3], series
    out = render_eer_plot(d)
    print(f"[plot_metrics.py] self-test OK, wrote {out}")


if __name__ == "__main__":
    main()
