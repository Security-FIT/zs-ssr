"""Naturalness of the reconstructed speech per (arch, run): UTMOS22 predicted MOS.

The reconstruction counterpart of vc_naturalness.py. The test utterances of one synthesizer are
reconstructed with a trained reconstructor and each output is scored with UTMOS22
(`evaluation/utmos.py`). `--limit` defaults to 0, the whole subset, because the mean is only
comparable with the VC input's on the same utterances.

Training does not keep the reconstructions, so they are synthesized again: `LOADERS` builds each
architecture's model and reconstruct_fn from its latest checkpoint and `EvalSet` supplies the
utterances. One (arch, run) per call, GPU required. Each call writes a one-row
`analysis/results/recon_naturalness_<arch>_<run>.tsv`.

    python -m analysis.recon_naturalness --arch freevc --synth 2 \
        --slt-root ../datasets/slt-sstc --librispeech-root ../datasets/LibriSpeech
    # an -align run: point --out-dir at its checkpoints (the embedding subset stays Test-<synth>)
    python -m analysis.recon_naturalness --arch freevc --synth 1 --out-dir outputs/freevc/synth-1-align
"""

from __future__ import annotations

import argparse
import glob
import os
from functools import partial

import numpy as np
import torch

from analysis.vc_naturalness import COLS, summarize
from dataset.sltsstc_tar import activate as activate_corpus
from evaluation.reconstruction_eval import EvalSet
from evaluation.utmos import load_utmos22, score_utmos22

_SR = 16000  # reconstruct_fn returns 16 kHz audio, the rate UTMOS22 expects


def latest_checkpoint(out_dir: str, pattern: str) -> str:
    """Most recently written checkpoint in `out_dir` matching `pattern`.

    Sorted by modification time because a directory can mix `_ssr_epoch<E>.pth`, `_ssr_last.pth`
    and sample-count `_ssr_<samples>.pth` names, which do not sort by training progress.
    """
    fs = glob.glob(os.path.join(out_dir, pattern))
    if not fs:
        raise FileNotFoundError(f"no checkpoint matching {pattern!r} in {out_dir}")
    return max(fs, key=os.path.getmtime)


def load_freevc(out_dir, emb_dim, device, args):
    """Build FreeVC + load its latest checkpoint. Returns (model, ckpt_path, reconstruct_fn)."""
    from freevc.ssl import build_ssl_backend
    from freevc.validation import reconstruct
    from train_freevc import build_model
    ckpt_path = latest_checkpoint(out_dir, "freevc_ssr_*.pth")
    ssl = build_ssl_backend("wavlm", device=device)
    model, _ = build_model(ssl.ssl_dim, emb_dim, device, ssl_backend=ssl)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.eval()
    return model, ckpt_path, reconstruct


def load_againvc(out_dir, emb_dim, device, args):
    """Build AGAIN-VC + load its latest checkpoint. Returns (model, ckpt_path, reconstruct_fn)."""
    from againvc.model import AgainReconstructor
    from againvc.reconstruct import reconstruct
    ckpt_path = latest_checkpoint(out_dir, "againvc_ssr_*.pth")
    model = AgainReconstructor(emb_dim=emb_dim).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.eval()
    return model, ckpt_path, reconstruct


def load_diffvc(out_dir, emb_dim, device, args):
    """Build DiffVC + load its latest checkpoint. Returns (model, ckpt_path, reconstruct_fn)."""
    from diffvc.reconstruct import reconstruct
    from diffvc.reconstructor import build_model
    ckpt_path = latest_checkpoint(out_dir, "diffvc_ssr_*.pth")
    model = build_model(emb_dim, device, args.enc_ckpt)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    model.eval()
    # the sampler settings are fixed here so reconstruct_fn keeps the shared signature
    recon = partial(reconstruct, n_timesteps=args.n_timesteps, mode=args.diff_mode)
    return model, ckpt_path, recon


# arch -> (checkpoint output root, loader)
LOADERS = {
    "freevc": ("outputs/freevc", load_freevc),
    "againvc": ("outputs/againvc", load_againvc),
    "diffvc": ("outputs/diffvc", load_diffvc),
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", default="freevc", choices=sorted(LOADERS))
    p.add_argument("--synth", type=int, required=True, help="test synthesizer number (Test-<n>)")
    p.add_argument("--split", default="test", choices=["dev", "test"])
    p.add_argument("--emb-dir", default="embeddings/non-finetuned/organizer-pretrained/exported_embeddings")
    p.add_argument("--slt-root", default="../datasets/slt-sstc")
    p.add_argument("--librispeech-root", default="../datasets/LibriSpeech")
    p.add_argument("--out-dir", default=None,
                   help="checkpoint dir (default outputs/<arch>/synth-<n>, the -align dir for aligned runs)")
    p.add_argument("--limit", type=int, default=0,
                   help="utterances reconstructed and scored, 0 for all (default, the same "
                        "utterances as the VC input table so the two means are comparable)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default=None,
                   help="default analysis/results/recon_naturalness_<arch>_<run>.tsv")
    # DiffVC only (load_diffvc)
    p.add_argument("--enc-ckpt", default="diffvc/checkpts/enc.pt")
    p.add_argument("--n-timesteps", type=int, default=30)
    p.add_argument("--diff-mode", default="ml", choices=["pf", "em", "ml"])
    args = p.parse_args()

    assert torch.cuda.is_available(), "GPU required, run this in a GPU job"
    device = "cuda"
    activate_corpus(args.slt_root)
    output_root, loader = LOADERS[args.arch]
    out_dir = args.out_dir or os.path.join(output_root, f"synth-{args.synth}")
    run = os.path.basename(out_dir.rstrip("/"))  # e.g. synth-1-align, names the output TSV

    synth_tag = f"{'Dev' if args.split == 'dev' else 'Test'}-{args.synth}"
    npz = os.path.join(args.emb_dir, f"{args.split}_embeddings.npz")
    eval_set = EvalSet([npz], args.slt_root, args.librispeech_root, normalize=True,
                       synthesizers=[synth_tag])
    assert eval_set.rows, f"no utterances for {synth_tag} in {npz}"
    emb_dim = int(eval_set.rows[0]["g"].shape[-1])  # the reconstructor is built for this g size

    model, ckpt_path, reconstruct_fn = loader(out_dir, emb_dim, device, args)
    m_22 = load_utmos22(device)

    rng = np.random.default_rng(args.seed)
    rows = eval_set.rows
    if args.limit and args.limit < len(rows):
        rows = [rows[i] for i in rng.choice(len(rows), size=args.limit, replace=False)]
    print(f"[recon_naturalness.py] {args.arch} {run} {synth_tag} | ckpt {os.path.basename(ckpt_path)} "
          f"| reconstructing {len(rows)} utterances")

    u22 = []
    for r in rows:
        wav = reconstruct_fn(model, r["vc_file"], r["g"], device, sr=_SR)
        b = score_utmos22(m_22, wav, device)
        if not np.isnan(b):  # NaN for clips too short to score
            u22.append(b)
    row = summarize(np.array(u22))
    print(f"[recon_naturalness.py] {args.arch} {run}: n={row['n']}  UTMOS22 median={row['utmos22_median']:.2f} "
          f"mean={row['utmos22_mean']:.2f} "
          f"(95%CI {row['utmos22_ci95_lo']:.2f}-{row['utmos22_ci95_hi']:.2f})")

    out = args.out or os.path.join("analysis", "results", f"recon_naturalness_{args.arch}_{run}.tsv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("\t".join(["arch", "run", "synth", "split"] + COLS) + "\n")
        vals = [args.arch, run, args.synth, args.split] + \
               [f"{row[c]:.3f}" if isinstance(row[c], float) else row[c] for c in COLS]
        f.write("\t".join(str(x) for x in vals) + "\n")
    print(f"[recon_naturalness.py] wrote {out}")


if __name__ == "__main__":
    main()
