"""Shared parts of the train_<model>.py scripts: the common command line and the in-loop validation.

Every reconstructor trains on the same (VC sample, LibriSpeech original, g) pairs and is
validated the same way: reconstruct dev utterances after each epoch, score them with the
ECAPA-TDNN ASV against real LibriSpeech utterances and log the EER to
`<out-dir>/val_metrics.txt` (redrawing `<out-dir>/val_eer.svg`). After training, dev and test
are scored once more, and the test run also writes its trials, the reconstructions' ASV
embeddings and the listening examples. A trainer only adds its own flags and passes its
`reconstruct_fn`.
"""

from __future__ import annotations

import argparse
import os

from evaluation.asv import ASVModel
from evaluation.plot_metrics import render_eer_plot
from evaluation.reconstruction_eval import EvalSet, run_asv_validation


def add_common_args(p: argparse.ArgumentParser, batch_size: int, align_max_lag: int) -> None:
    """Flags every trainer takes. `batch_size` and `align_max_lag` are the per-model defaults."""
    p.add_argument("--exp-npz", nargs="+", required=True, help="train_embeddings.npz path(s); drives the data")
    p.add_argument("--slt-root", default="../datasets/slt-sstc", help="root for VC samples (npz file_paths are relative to it)")
    p.add_argument("--librispeech-root", default="../datasets/LibriSpeech", help="root with <subset>/<reader>/<chapter>/*.flac targets")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=batch_size)
    p.add_argument("--num-workers", type=int, default=4, help="dataloader workers")
    p.add_argument("--log-interval", type=int, default=50, help="optimizer steps between progress-bar loss updates")
    p.add_argument("--max-steps", type=int, default=None,
                   help="stop after N optimizer steps and save <model>_ssr_last.pth; also the "
                        "--overfit-one-batch length (default 200)")
    p.add_argument("--overfit-one-batch", action="store_true",
                   help="repeat one batch for --max-steps steps, a check that the pipeline learns")
    p.add_argument("--limit-samples", type=int, default=None,
                   help="cap the number of (VC, LibriSpeech) pairs built (fast startup for smoke tests)")
    p.add_argument("--resume", default=None, help="checkpoint .pth to resume from")
    p.add_argument("--align", action="store_true",
                   help="correct a constant VC/source frame offset per utterance before cropping, for "
                        "synthesizers whose VC output is time-shifted (Train-1, Train-4; see duration_check.py)")
    p.add_argument("--align-max-lag", type=int, default=align_max_lag,
                   help="max offset searched by --align, in the model's frames (+/-)")
    # VC subsets: the npz hold all of them and are filtered by their 'synthesizer' field
    p.add_argument("--train-synth", nargs="+", default=None, help="VC subset(s) to train on, e.g. Train-3. Default: all.")
    p.add_argument("--dev-synth", nargs="+", default=None, help="VC subset(s) for dev validation, e.g. Dev-3. Default: all.")
    p.add_argument("--test-synth", nargs="+", default=None, help="VC subset(s) for the final test ASV, e.g. Test-3. Default: all.")
    # validation: reconstruction vs real LibriSpeech, scored by the ECAPA-TDNN ASV
    p.add_argument("--dev-npz", nargs="+", default=None, help="dev_embeddings.npz path(s): enables end-of-epoch ASV validation")
    p.add_argument("--test-npz", nargs="+", default=None, help="test_embeddings.npz path(s): final ASV after training")
    p.add_argument("--val-samples", type=int, default=0, help="dev utterances reconstructed and scored per validation (<=0 = all)")
    p.add_argument("--test-samples", type=int, default=0, help="test utterances scored in the final ASV (<=0 = all)")
    p.add_argument("--asv-model", default="evaluation/models/wespeaker/ecapa-tdnn1024_lm", help="WeSpeaker ECAPA-TDNN model dir")
    p.add_argument("--asv-device", default="cuda", help="device for the ASV (and --wer ASR) model")
    p.add_argument("--wer", action="store_true",
                   help="also measure the word error rate (Whisper ASR of the reconstruction vs the "
                        "LibriSpeech transcript) during validation and the final eval")
    p.add_argument("--asr-model", default="base.en",
                   help="Whisper model for --wer (base.en / small.en / medium.en; caches to ~/.cache/whisper)")
    p.add_argument("--normalize-emb", action=argparse.BooleanOptionalAction, default=True,
                   help="L2-normalize the source-speaker embeddings used as g (default on)")
    p.add_argument("--average-emb", action=argparse.BooleanOptionalAction, default=True,
                   help="g = per-speaker mean embedding (default) or the per-utterance one (--no-average-emb); "
                        "dev/test always use the per-utterance embedding")


def smoke_limit(args) -> int | None:
    """Number of pairs to build: --limit-samples, else a few batches for --overfit-one-batch, else all."""
    return args.limit_samples or (max(args.batch_size * 4, 64) if args.overfit_one_batch else None)


class Validator:
    """In-loop ASV validation of one model, with the ASV/ASR models loaded on first use.

    `reconstruct_fn(model, vc_wav, g, device, sr=...) -> 1-D wav` is the model's reconstruction
    (see evaluation/reconstruction_eval.py). `meta` goes into the listening-examples manifest.
    """

    def __init__(self, args, model, device: str, reconstruct_fn, meta: dict):
        self.args, self.model, self.device = args, model, device
        self.reconstruct_fn, self.meta = reconstruct_fn, meta
        # built up front so a LibriSpeech coverage problem fails before training
        self.dev_eval = (EvalSet(args.dev_npz, args.slt_root, args.librispeech_root,
                                 normalize=args.normalize_emb, synthesizers=args.dev_synth)
                         if args.dev_npz else None)
        self._asv = self._asr = None

    def asv(self) -> ASVModel:
        if self._asv is None:
            print(f"[train_common.py] loading ASV model {self.args.asv_model} on {self.args.asv_device}")
            self._asv = ASVModel(self.args.asv_model, device=self.args.asv_device)
        return self._asv

    def asr(self):
        if not self.args.wer:
            return None
        if self._asr is None:
            from evaluation.asr import ASRModel
            print(f"[train_common.py] loading ASR model {self.args.asr_model} on {self.args.asv_device} (WER)")
            self._asr = ASRModel(self.args.asr_model, device=self.args.asv_device)
        return self._asr

    def validate(self, eval_set, n: int, tag: str, epoch: int, step: int, save_trials: bool = False):
        """Score `n` utterances of `eval_set` and append one line to val_metrics.txt.

        With `save_trials`, the trials and the reconstructions' ASV embeddings are also written,
        named by epoch so a later resumed run doesn't overwrite them.
        """
        out_dir = self.args.out_dir
        trials_out = os.path.join(out_dir, f"{tag}_trials_epoch{epoch}.tsv") if save_trials else None
        emb_out = os.path.join(out_dir, f"{tag}_recon_embeddings_epoch{epoch}.npz") if save_trials else None
        res = run_asv_validation(self.model, self.asv(), eval_set, self.device, n, out_dir,
                                 reconstruct_fn=self.reconstruct_fn, trials_out=trials_out,
                                 emb_out=emb_out, asr=self.asr())
        line = (f"epoch {epoch} samples {step * self.args.batch_size} {tag}: EER={res['eer']:.4f} "
                f"(n={res['n']} mean_tgt={res['mean_target']:.3f} mean_non={res['mean_nontarget']:.3f})")
        if res["n_wer"]:
            line += f" WER={res['wer']:.4f}"
        print(f"[train_common.py] {line}")
        with open(os.path.join(out_dir, "val_metrics.txt"), "a") as f:
            f.write(line + "\n")
        try:  # a plotting error must not stop training
            render_eer_plot(out_dir)
        except Exception as e:
            print(f"[train_common.py] failed to update the EER plot: {e}")

    def dev_epoch(self, epoch: int, step: int):
        """End-of-epoch dev validation (no-op without --dev-npz)."""
        if self.dev_eval is not None:
            self.validate(self.dev_eval, self.args.val_samples, f"dev-epoch{epoch}", epoch, step)

    def final_eval(self, epoch: int, step: int):
        """Dev and test after training. Test also writes the trials, embeddings and examples."""
        args = self.args
        if self.dev_eval is not None:
            self.validate(self.dev_eval, args.val_samples, "dev-final", epoch, step)
        if not args.test_npz:
            return
        test_eval = EvalSet(args.test_npz, args.slt_root, args.librispeech_root,
                            normalize=args.normalize_emb, synthesizers=args.test_synth)
        self.validate(test_eval, args.test_samples, "test-final", epoch, step, save_trials=True)
        try:  # a finished run must not fail on the examples
            from evaluation.examples import generate_examples
            generate_examples(self.model, self.asv(), test_eval, self.device, args.out_dir,
                              reconstruct_fn=self.reconstruct_fn,
                              meta={**self.meta, "split": "test", "test_synth": args.test_synth,
                                    "epoch": epoch})
        except Exception as e:
            print(f"[train_common.py] failed to save examples: {e}")


def main():
    """Self-test without models: the flags parse and validate() writes the val_metrics.txt line."""
    import sys
    import tempfile

    p = argparse.ArgumentParser()
    add_common_args(p, batch_size=8, align_max_lag=50)
    with tempfile.TemporaryDirectory() as d:
        args = p.parse_args(["--exp-npz", "x.npz", "--out-dir", d, "--overfit-one-batch"])
        assert smoke_limit(args) == 64 and args.num_workers == 4

        def fake_asv_validation(model, asv, eval_set, device, n, out_dir, **kw):
            assert kw["trials_out"] == os.path.join(d, "test-final_trials_epoch3.tsv"), kw
            return {"eer": 0.25, "n": n, "mean_target": 0.5, "mean_nontarget": 0.1, "n_wer": 0}

        module = sys.modules[__name__]
        module.run_asv_validation, module.render_eer_plot = fake_asv_validation, lambda out_dir: None
        val = Validator(args, model=None, device="cpu", reconstruct_fn=None, meta={})
        val._asv = object()
        val.validate(None, 10, "test-final", epoch=3, step=5, save_trials=True)
        with open(os.path.join(d, "val_metrics.txt")) as f:
            line = f.read()
        assert line == ("epoch 3 samples 40 test-final: EER=0.2500 "
                        "(n=10 mean_tgt=0.500 mean_non=0.100)\n"), line
    print("[train_common.py] OK")


if __name__ == "__main__":
    main()
