"""Quality of the SLT-SSTC test VC samples themselves: target-speaker similarity and WER.

Nothing here touches a reconstructor. It measures how good each voice conversion is, so that
`correlate_quality.py` can relate it to how well the sample reconstructs. For every VC wav
`id10xxx-<video>-<segment>-<reader>-<chapter>-<utt>.wav`:

* target similarity: ECAPA-TDNN cosine (`evaluation/asv.py`) between the VC wav and one random
  VoxCeleb1 utterance of `id10xxx`, the voice it was converted into. The clip the conversion was
  conditioned on (`vox1/wav/id10xxx/<video>/<segment>.wav`) is never picked.
* nontarget similarity: the same VC embedding against one random utterance of another VoxCeleb1
  speaker. With the target scores this gives a target-speaker EER per synthesizer, lower means
  the VC output passes better as the target voice.
* WER: Whisper transcript of the VC wav against the LibriSpeech transcript of its source
  utterance (`evaluation/asr.py`).

One `--synth Test-N` per run. ASR takes hours and the similarities minutes, so `--skip-wer`
recomputes only the similarity columns and copies the WER columns from the TSV already on disk.
Output: `<out-dir>/vc_quality_<synth>.tsv`, one row per VC utterance, read by
`plot_main_results.py`, `correlate_quality.py`, `tail_table.py` and `vc_naturalness.py`.

    python -m analysis.vc_quality --synth Test-3 ...
"""

from __future__ import annotations

import argparse
import csv
import functools
import glob
import os

import numpy as np

WER_COLS = ["wer", "wer_edits", "wer_nref", "ref_text", "hyp_text"]


def clean_text(s: str) -> str:
    """Collapse whitespace so a transcript fits in one TSV cell."""
    return " ".join((s or "").split())


def list_synth_files(test_npz: str, synth: str) -> list[str]:
    """Dataset-relative VC wav paths of one synthesizer, from the test protocol npz."""
    d = np.load(test_npz, allow_pickle=True)
    return [str(fp) for fp, sy in zip(d["file_paths"], d["synthesizer"]) if str(sy) == synth]


def vox_reference(vc_path: str) -> tuple[str, str, str]:
    """(target id, video, segment) of the VoxCeleb clip the conversion was conditioned on.

    Parsed from the right like `dataset.resolver.source_utt_id`, since the last three fields
    are always reader-chapter-utterance and a video id may contain hyphens
    (`id10005-1geDB-I2TjE-00001-2094-142345-0058` -> id10005, `1geDB-I2TjE`, `00001`).
    """
    p = os.path.splitext(os.path.basename(vc_path))[0].split("-")
    return p[0], "-".join(p[1:-4]), p[-4]


@functools.lru_cache(maxsize=None)
def speaker_clips(wav_root: str, spk: str) -> list[str]:
    """All VoxCeleb1 wavs of one speaker, sorted (globbed once per speaker)."""
    return sorted(glob.glob(os.path.join(wav_root, spk, "*", "*.wav")))


def existing_wer_columns(out_tsv: str) -> dict[str, list[str]]:
    """file_path -> WER_COLS values from a previous run's TSV, for `--skip-wer`.

    Read with QUOTE_NONE because the TSV is written without escaping: a Whisper hypothesis that
    starts with `"` would otherwise swallow the following lines.
    """
    if not os.path.exists(out_tsv):
        print(f"[vc_quality.py] --skip-wer: no existing {out_tsv}, the WER columns stay empty")
        return {}
    with open(out_tsv) as f:
        rows = {r["file_path"]: [r.get(c, "") for c in WER_COLS]
                for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)}
    print(f"[vc_quality.py] --skip-wer: copying the WER columns of {len(rows)} utterances from {out_tsv}")
    return rows


def analyze(args):
    """Score every VC wav of `args.synth` and write vc_quality_<synth>.tsv."""
    from tqdm import tqdm

    from dataset import sltsstc_tar
    from dataset.resolver import LibriSpeechResolver, source_utt_id
    from evaluation.asr import ASRModel, librispeech_transcript, word_error_rate
    from evaluation.asv import ASVModel, compute_eer

    sltsstc_tar.activate(args.slt_root)
    asv = ASVModel(args.asv_model, device=args.device)
    # the transcript models are only needed when the WER is computed
    resolver = None if args.skip_wer else LibriSpeechResolver(args.librispeech_root)
    asr = None if args.skip_wer else ASRModel(args.asr_model, device=args.device)
    wav_root = os.path.join(args.vox_root, "vox1", "wav")
    speakers = sorted(os.listdir(wav_root))  # VoxCeleb1 ids, the nontarget pool

    fps = list_synth_files(args.test_npz, args.synth)
    if args.limit:
        fps = fps[: args.limit]
    print(f"[vc_quality.py] {args.synth}: {len(fps)} VC utterances, {len(speakers)} VoxCeleb1 speakers")
    rng = np.random.default_rng(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    out_tsv = os.path.join(args.out_dir, f"vc_quality_{args.synth}.tsv")
    # read the old WER columns before the file is truncated for writing
    old_wer = existing_wer_columns(out_tsv) if args.skip_wer else {}
    # each trial is stored as the clip scored against and its cosine
    cols = ["file_path", "synthesizer", "target_id", "source_utt",
            "target_clip", "target_sim", "nontarget_id", "nontarget_clip", "nontarget_sim"] + WER_COLS

    sims, nonsims = [], []                 # target and nontarget cosines, for the EER
    wer_edits, wer_nref, n_wer = 0, 0, 0   # summed over the utterances with a transcript
    n_unresolved = 0                       # no LibriSpeech original found
    n_unexcluded = 0                       # conditioning clip not found, so nothing excluded
    with open(out_tsv, "w") as f:
        f.write("\t".join(cols) + "\n")
        for fp in tqdm(fps, desc=f"vc_quality {args.synth}"):
            tid, video, segment = vox_reference(fp)
            sid = source_utt_id(fp)
            wav = sltsstc_tar.read_wav(os.path.join(args.slt_root, fp), sr=16000)
            emb = asv.embed_wav(wav, 16000)

            # target trial: a random utterance of the target speaker other than the clip this
            # conversion was conditioned on, which would inflate the score
            used = os.path.join(wav_root, tid, video, segment + ".wav")
            pool = speaker_clips(wav_root, tid)
            if used not in pool:
                n_unexcluded += 1
            pool = [w for w in pool if w != used]
            if not pool:
                raise RuntimeError(f"target {tid} has no clip besides its reference {used}")
            target_clip = pool[int(rng.integers(len(pool)))]
            sim = asv.cosine(emb, asv.embed_file(target_clip))
            sims.append(sim)

            # nontarget trial: a random utterance of a random other VoxCeleb1 speaker
            nontarget_id = tid
            while nontarget_id == tid:
                nontarget_id = speakers[int(rng.integers(len(speakers)))]
            nontarget_pool = speaker_clips(wav_root, nontarget_id)
            nontarget_clip = nontarget_pool[int(rng.integers(len(nontarget_pool)))]
            nonsim = asv.cosine(emb, asv.embed_file(nontarget_clip))
            nonsims.append(nonsim)

            if args.skip_wer:
                wer_cols = old_wer.get(fp, [""] * len(WER_COLS))
            else:
                original = resolver.path(sid)
                if original is None:  # no LibriSpeech original, so no reference transcript
                    n_unresolved += 1
                    wer_cols = [""] * len(WER_COLS)
                else:
                    ref = librispeech_transcript(original)
                    hyp = asr.transcribe(wav, 16000)
                    e, nref = word_error_rate(ref, hyp, normalizer=asr.normalize)
                    wer_edits += e
                    wer_nref += nref
                    n_wer += 1
                    uw = e / nref if nref else float("nan")  # this utterance's own WER
                    wer_cols = [f"{uw:.4f}", str(e), str(nref), clean_text(ref), clean_text(hyp)]

            f.write("\t".join([
                fp, args.synth, tid, sid,
                os.path.relpath(target_clip, wav_root), f"{sim:.6f}",
                nontarget_id, os.path.relpath(nontarget_clip, wav_root), f"{nonsim:.6f}",
            ] + wer_cols) + "\n")

    eer = compute_eer(sims, nonsims)
    # unexcluded_refs should be 0, otherwise some file names did not match a VoxCeleb clip
    summary = (f"[vc_quality.py] {args.synth}: n={len(fps)} "
               f"mean_target_sim={np.mean(sims):.4f} median={np.median(sims):.4f} "
               f"EER={eer:.4f} (mean_non={np.mean(nonsims):.4f}) unexcluded_refs={n_unexcluded}")
    if args.skip_wer:
        summary += f" [--skip-wer: {len(old_wer)} WER rows copied]"
    else:
        micro_wer = wer_edits / wer_nref if wer_nref else float("nan")  # total edits / total words
        summary += f" micro_WER={micro_wer:.4f} (n_wer={n_wer}, unresolved={n_unresolved})"
    print(summary)
    print(f"[vc_quality.py] wrote {out_tsv}")


def self_test():
    """File name parsing and transcript cleaning, no models or audio."""
    assert clean_text("  a\tb\nc ") == "a b c"
    fp = "test/Test-3/id10270-5r0dWxy17C8-00001-1462-170142-0034.wav"
    assert vox_reference(fp) == ("id10270", "5r0dWxy17C8", "00001"), vox_reference(fp)
    # a hyphenated video id
    assert vox_reference("id10005-1geDB-I2TjE-00001-2094-142345-0058.wav") == \
        ("id10005", "1geDB-I2TjE", "00001")
    # the `_gen` suffix of Test-5/6 only touches the utterance field
    assert vox_reference("id10270-GWXujl-xAVM-00041-3752-4943-0020_gen.wav") == \
        ("id10270", "GWXujl-xAVM", "00041")
    print("[vc_quality.py] self-test OK")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-npz",
                   default="embeddings/non-finetuned/organizer-pretrained/exported_embeddings/test_embeddings.npz")
    p.add_argument("--synth", help="one synthesizer subset, e.g. Test-3 (required unless --self-test)")
    p.add_argument("--slt-root", default="../datasets/slt-sstc")
    p.add_argument("--librispeech-root", default="../datasets/LibriSpeech")
    p.add_argument("--vox-root", default="../datasets/voxceleb")
    p.add_argument("--asv-model",
                   default="evaluation/models/wespeaker/ecapa-tdnn1024_lm")
    p.add_argument("--asr-model", default="base.en")
    p.add_argument("--skip-wer", action="store_true",
                   help="recompute only the similarity columns (no Whisper, minutes instead of "
                        "hours) and copy the WER columns from the TSV already on disk")
    p.add_argument("--out-dir", default="analysis/results")
    p.add_argument("--seed", type=int, default=1234, help="trial sampling seed")
    p.add_argument("--limit", type=int, default=0, help="score only the first N utterances (smoke test)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--self-test", action="store_true", help="run the self-test and exit")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.synth:
        p.error("--synth is required (e.g. --synth Test-3)")
    analyze(args)


if __name__ == "__main__":
    main()
