"""Shared evaluation harness for the source-speech reconstructors.

- `asv`                 : ASVModel (WeSpeaker ECAPA-TDNN) + compute_eer.
- `reconstruction_eval` : EvalSet + run_asv_validation(reconstruct_fn=...).
- `examples`            : generate_examples(reconstruct_fn=...) (listening test).
- `plot_metrics`        : parse_val_metrics + render_eer_plot (live EER plot).
- `asr`                 : Whisper transcription + WER (`--wer`).
- `utmos`               : UTMOS22 predicted MOS (naturalness, used by `analysis/`).
- `models/`             : the ASV model weights (WeSpeaker), referenced by path.

Import submodules directly (e.g. `from evaluation.asv import ASVModel`) so a plain
`import evaluation` doesn't pull in wespeaker/librosa. Every reconstructor plugs its own
`reconstruct_fn` into this harness.
"""
