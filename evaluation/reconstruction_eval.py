"""In-loop ASV validation of the source-speech reconstructors.

Held-out (dev or test) VC samples are reconstructed and compared with real audio of their
source speaker by an ECAPA-TDNN speaker verifier (WeSpeaker, `evaluation/asv.py`). For a
reconstruction of source speaker S:

* positive trial: the reconstruction against a random real LibriSpeech utterance of S
* negative trial: the reconstruction against a random real LibriSpeech utterance of another speaker

The EER over the positive and negative cosine scores says whether the reconstructions recover
the source speaker. Trials are sampled in memory from the exported dev/test npz (the source
speaker is the `reader` field of the file name), so no trial lists are needed.

`run_asv_validation` takes a `reconstruct_fn(model, vc_wav, g, device, sr=...) -> 1-D 16 kHz wav`,
through which every reconstructor plugs in its own inference.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from tqdm import tqdm

from dataset.pairing import _maybe_norm
from dataset.resolver import LibriSpeechResolver, source_utt_id
from evaluation.asr import librispeech_transcript, word_error_rate
from evaluation.asv import ASVModel, compute_eer


class EvalSet:
    """Held-out utterances and a per-speaker pool of real LibriSpeech references.

    `rows` holds one dict per utterance: its VC wav, the per-utterance source embedding `g`
    (as available at inference), the source speaker and its LibriSpeech original.
    `refs[speaker]` lists every original of that speaker, the pool the trial references are
    drawn from.
    """

    def __init__(self, npz_paths, slt_root: str, librispeech_root: str, normalize: bool = True,
                 synthesizers: list[str] | None = None):
        resolver = LibriSpeechResolver(librispeech_root)
        synth_filter = set(synthesizers) if synthesizers else None
        self.rows: list[dict] = []            # {vc_file, g, speaker, original}
        self.refs: dict[str, list[str]] = {}  # speaker -> LibriSpeech original paths
        n_total = n_missing = 0
        for npz_path in npz_paths:
            d = np.load(npz_path, allow_pickle=True)
            fps, embs = d["file_paths"], d["embeddings"]
            syn = d["synthesizer"] if "synthesizer" in d.files else [None] * len(fps)
            for fp, e, sy in tqdm(zip(fps, embs, syn), total=len(fps), desc=f"eval-resolve {os.path.basename(npz_path)}"):
                if synth_filter is not None and str(sy) not in synth_filter:
                    continue
                n_total += 1
                fp = str(fp)
                spk = source_utt_id(fp).split("-")[0]  # reader = source speaker
                original = resolver.path(source_utt_id(fp))
                if original is None:
                    n_missing += 1
                    continue
                g = _maybe_norm(np.asarray(e, dtype=np.float64), normalize).astype(np.float32)
                vc_file = os.path.join(slt_root, fp)
                self.rows.append({"vc_file": vc_file, "g": g, "speaker": spk, "original": original})
                self.refs.setdefault(spk, []).append(original)
        self.speakers = sorted(self.refs)
        print(
            f"[reconstruction_eval.py] EvalSet: {len(self.rows)}/{n_total} utterances resolved to "
            f"LibriSpeech ({n_missing} missing), {len(self.speakers)} source speakers"
        )
        if n_missing > 0:  # every trial needs real audio of the source speaker
            raise RuntimeError(
                f"[reconstruction_eval.py] {n_missing}/{n_total} dev/test utterances have no "
                "LibriSpeech original. Check that --librispeech-root covers all dev/test source speakers."
            )


def run_asv_validation(model, asv: ASVModel, eval_set: EvalSet, device: str,
                       n_samples: int | None, out_dir: str, reconstruct_fn,
                       seed: int = 1234, sr: int = 16000,
                       trials_out: str | None = None, emb_out: str | None = None,
                       asr=None) -> dict:
    """Reconstruct up to `n_samples` utterances of `eval_set`, score them and return the EER.

    Each probe of speaker S gets a positive reference (a random utterance of S other than the
    probe's own original, probes without one are skipped) and a negative reference (a random
    utterance of another speaker). The same `seed` picks the same probes and references on
    every call, so the EER is comparable across epochs.

    With `asr` (an `evaluation.asr.ASRModel`) each reconstruction is also transcribed and the
    corpus WER is returned in `wer` (NaN without `asr`). `trials_out` writes every trial to a
    TSV, `emb_out` saves the reconstruction embeddings with their trials to an npz, so trials
    can be rescored later without reconstructing again.
    """
    rng = np.random.default_rng(seed)
    rows = eval_set.rows
    if n_samples and n_samples > 0 and n_samples < len(rows):
        idx = rng.choice(len(rows), size=n_samples, replace=False)
        rows = [rows[i] for i in idx]

    was_training = model.training
    model.eval()
    # cudnn.benchmark (enabled by Coqui's trainer) re-autotunes for every new input length,
    # ~20 s per full variable-length utterance. Off during validation, restored afterwards.
    cudnn_bench = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    target, nontarget = [], []  # positive and negative cosine scores
    trials = [] if trials_out else None
    # row-aligned columns of the emb_out npz
    emb = {k: [] for k in ("recon_emb", "vc_file", "source_speaker", "target_original",
                           "pos_ref", "pos_score", "neg_ref", "neg_speaker", "neg_score",
                           "utt_wer", "ref_text", "hyp_text")} if emb_out else None
    n_skipped = 0
    wer_edits = wer_refs = wer_n = 0  # summed over the transcribed probes
    other_speakers = eval_set.speakers
    try:
        for r in tqdm(rows, desc="asv-val"):
            spk = r["speaker"]

            # positive: another real utterance of the same speaker. Scoring against the probe's
            # own original would compare the reconstruction with its training target.
            pos_pool = [p for p in eval_set.refs[spk] if p != r["original"]]
            if not pos_pool:
                n_skipped += 1
                continue
            pos = pos_pool[int(rng.integers(len(pos_pool)))]

            # negative: a real utterance of a different speaker (the same one if it is the only speaker)
            neg_spk = spk
            while neg_spk == spk and len(other_speakers) > 1:
                neg_spk = other_speakers[int(rng.integers(len(other_speakers)))]
            neg = eval_set.refs[neg_spk][int(rng.integers(len(eval_set.refs[neg_spk])))]

            wav = reconstruct_fn(model, r["vc_file"], r["g"], device, sr=sr)
            recon_emb = asv.embed_wav(wav, sr)
            ts = asv.cosine(recon_emb, asv.embed_file(pos))
            ns = asv.cosine(recon_emb, asv.embed_file(neg))
            target.append(ts)
            nontarget.append(ns)

            # WER of the same reconstruction against the LibriSpeech transcript
            uwer = ref = hyp = None
            if asr is not None:
                ref = librispeech_transcript(r["original"])
                hyp = asr.transcribe(wav, sr)
                e, nref = word_error_rate(ref, hyp, normalizer=asr.normalize)
                wer_edits += e
                wer_refs += nref
                wer_n += 1
                uwer = e / nref if nref else float("nan")  # this utterance's own WER

            if trials is not None:
                row = [r["vc_file"], spk, r["original"],
                       pos, f"{ts:.6f}", neg, neg_spk, f"{ns:.6f}"]
                if asr is not None:
                    row += [f"{uwer:.4f}", ref, hyp]
                trials.append(tuple(row))

            if emb is not None:
                emb["recon_emb"].append(np.asarray(recon_emb, dtype=np.float32).ravel())
                emb["vc_file"].append(r["vc_file"])
                emb["source_speaker"].append(spk)
                emb["target_original"].append(r["original"])
                emb["pos_ref"].append(pos)
                emb["pos_score"].append(ts)
                emb["neg_ref"].append(neg)
                emb["neg_speaker"].append(neg_spk)
                emb["neg_score"].append(ns)
                emb["utt_wer"].append(uwer if asr is not None else float("nan"))
                emb["ref_text"].append(ref if asr is not None else "")
                emb["hyp_text"].append(hyp if asr is not None else "")
    finally:
        torch.backends.cudnn.benchmark = cudnn_bench
        if was_training:
            model.train()

    if n_skipped:
        print(f"[reconstruction_eval.py] skipped {n_skipped} probe(s) with no same-speaker "
              f"reference other than the probe's own original")
    if trials is not None:
        cols = ["recon_source_vc", "source_speaker", "target_original", "pos_ref", "pos_score",
                "neg_ref", "neg_speaker", "neg_score"]
        if asr is not None:
            cols += ["utt_wer", "ref_text", "hyp_text"]
        with open(trials_out, "w") as f:
            f.write("\t".join(cols) + "\n")
            for t in trials:
                f.write("\t".join(str(x) for x in t) + "\n")
        print(f"[reconstruction_eval.py] wrote {len(trials)} trials to {trials_out}")

    if emb is not None and emb["recon_emb"]:
        np.savez(
            emb_out,
            recon_emb=np.stack(emb["recon_emb"]).astype(np.float32),
            **{k: np.asarray(emb[k]) for k in emb if k != "recon_emb"},
        )
        print(f"[reconstruction_eval.py] wrote {len(emb['recon_emb'])} reconstruction embeddings "
              f"and scores to {emb_out}")

    eer = compute_eer(target, nontarget)
    return {
        "eer": eer,
        "n": len(target),
        "n_skipped": n_skipped,
        "mean_target": float(np.mean(target)) if target else float("nan"),
        "mean_nontarget": float(np.mean(nontarget)) if nontarget else float("nan"),
        "wer": (wer_edits / wer_refs) if wer_refs else float("nan"),  # total edits / total words
        "n_wer": wer_n,
    }
