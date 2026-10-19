"""Train a zero-shot source-speech reconstructor (DiffVC).

The diffusion prior comes from the VC sample (DiffVC mel -> frozen LibriTTS "average voice"
encoder), the speaker is the source-speaker embedding `g`, and the diffusion decoder learns to
generate the mel of the original LibriSpeech utterance (score-matching loss). `use_ref_t=False`,
so the decoder is conditioned on `g` alone. Only the decoder is trained (see
`diffvc/reconstructor.py`). Based on DiffVC
(https://github.com/huawei-noah/Speech-Backbones/tree/main/DiffVC, MIT, see diffvc/LICENSE).
Validation turns the mel into a waveform with the vendored HiFi-GAN vocoder (`diffvc/vocoder.py`,
https://github.com/jik876/hifi-gan, MIT, see diffvc/THIRD_PARTY_NOTICE). Sampling is iterative,
so it uses a small `--n-timesteps`.

Requires the pretrained encoder `diffvc/checkpts/enc.pt` and the HiFi-GAN generator at
`diffvc/checkpts/vocoder/generator` (see README). The frozen encoder's outputs are cached to
`libritts_cache/` on the first run (`--no-libritts-cache` to disable).

Single GPU. Run it as a cluster job (scripts/sge/diffvc/).

Example:
    python train_diffvc.py \
        --exp-npz embeddings/non-finetuned/organizer-pretrained/exported_embeddings/train_embeddings.npz \
        --dev-npz .../dev_embeddings.npz --test-npz .../test_embeddings.npz \
        --train-synth Train-3 --dev-synth Dev-3 --test-synth Test-3 \
        --slt-root ../datasets/slt-sstc --librispeech-root ../datasets/LibriSpeech \
        --out-dir outputs/diffvc/synth-3 --batch-size 32 --max-steps 515000
"""

from __future__ import annotations

import argparse
import os
from functools import partial

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.npz_cache import NpzCache
from dataset.pairing import build_parallel_samples
from dataset.sltsstc_tar import activate as activate_corpus
from diffvc.dataset import ParallelMelReconDataset, collate, prime_encoder_cache
from diffvc.reconstruct import reconstruct as diffvc_reconstruct
from diffvc.reconstructor import build_model
from train_common import Validator, add_common_args, smoke_limit


def _save(model, optim, epoch, step, out_dir, tag):
    """Checkpoint as diffvc_ssr_<tag>.pth, tag = epoch<E> or last.

    Written via tmp+rename so a kill can't truncate an existing checkpoint.
    """
    path = os.path.join(out_dir, f"diffvc_ssr_{tag}.pth")
    torch.save({"model": model.state_dict(), "optim": optim.state_dict(),
                "epoch": epoch, "step": step}, path + ".tmp")
    os.replace(path + ".tmp", path)
    print(f"[train_diffvc.py] saved {path}")


def main():
    p = argparse.ArgumentParser()
    add_common_args(p, batch_size=32, align_max_lag=50)
    p.add_argument("--lr", type=float, default=1e-4, help="DiffVC decoder default")
    p.add_argument("--grad-norm", type=float, default=1.0, help="DiffVC default grad clip")
    p.add_argument("--train-frames", type=int, default=128, help="training mel-frame crop (DiffVC default)")
    p.add_argument("--enc-ckpt", default="diffvc/checkpts/enc.pt",
                   help="pretrained LibriTTS 'average voice' encoder (frozen)")
    p.add_argument("--libritts-cache", default="libritts_cache",
                   help="dir for the cached frozen-encoder outputs (filled on the first run)")
    p.add_argument("--no-libritts-cache", dest="libritts_cache", action="store_const", const=None,
                   help="disable the encoder cache; run the frozen encoder live each step")
    p.add_argument("--n-timesteps", type=int, default=6, help="diffusion reverse steps for validation sampling")
    p.add_argument("--diff-mode", default="ml", choices=["pf", "em", "ml"], help="diffusion solver for sampling")
    args = p.parse_args()

    assert torch.cuda.is_available(), "GPU required (run it as a cluster job)."
    device = "cuda"
    os.makedirs(args.out_dir, exist_ok=True)
    activate_corpus(args.slt_root)  # tar-packed slt-root? read VC wavs from the tars, else loose

    cache = args.libritts_cache
    limit = smoke_limit(args)
    samples = build_parallel_samples(
        args.exp_npz, args.slt_root, args.librispeech_root,
        average=args.average_emb, normalize=args.normalize_emb, limit=limit,
        synthesizers=args.train_synth,
    )
    assert samples, "no (VC, LibriSpeech) pairs found (check --slt-root / --librispeech-root vs --exp-npz)"
    emb_dim = samples[0][2].shape[-1]
    print(f"[train_diffvc.py] {len(samples)} (VC->LibriSpeech) pairs, g dim={emb_dim}, "
          f"normalize_emb={args.normalize_emb}, average_emb={args.average_emb}, cache={'on' if cache else 'off'}")

    print(f"[train_diffvc.py] building the DiffVC reconstructor; encoder <- {args.enc_ckpt}")
    model = build_model(emb_dim, device, args.enc_ckpt)

    mel_cache = None
    if cache:
        # computes the encoder outputs of subsets not cached yet (one npz per subset)
        prime_encoder_cache(model, args.exp_npz, args.slt_root, cache, device,
                            synthesizers=args.train_synth, limit=limit)
        mel_cache = NpzCache(cache, args.slt_root)  # looked up by the VC wav path
    train_ds = ParallelMelReconDataset(samples, train_frames=args.train_frames,
                                       align=args.align, align_max_lag=args.align_max_lag,
                                       mel_cache=mel_cache)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True, collate_fn=collate,
    )

    optim = torch.optim.Adam(model.decoder.parameters(), lr=args.lr)  # the encoder stays frozen

    start_epoch, global_step = 0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        start_epoch, global_step = ckpt["epoch"], ckpt["step"]
        print(f"[train_diffvc.py] resumed from {args.resume} @ epoch {start_epoch} step {global_step}")
    epochs_done = start_epoch  # completed epochs, used to name the validation results

    # validation runs the encoder live, and the sampling settings are bound into the reconstruct fn
    recon_fn = partial(diffvc_reconstruct, n_timesteps=args.n_timesteps, mode=args.diff_mode)
    val = Validator(args, model, device, recon_fn,
                    meta={"architecture": "diffvc", "n_timesteps": args.n_timesteps,
                          "diff_mode": args.diff_mode})

    def do_step(batch):
        content, target, lengths, g = (t.to(device, non_blocking=True) for t in batch)
        if cache:  # `content` is the cached encoder output, i.e. the diffusion prior mean
            loss = model.compute_loss(target, lengths, g, mean=content)
        else:  # `content` is the VC mel and compute_loss runs the frozen encoder on it
            loss = model.compute_loss(target, lengths, g, content=content)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), args.grad_norm)
        optim.step()
        return loss

    model.train()

    if args.overfit_one_batch:
        steps = args.max_steps or 200
        batch = next(iter(train_loader))
        print(f"[train_diffvc.py] overfit-one-batch: {steps} steps on one batch (B={batch[0].size(0)})")
        pbar = tqdm(range(steps), desc="overfit")
        for step in pbar:
            loss = do_step(batch)
            if step % args.log_interval == 0 or step == steps - 1:
                pbar.set_postfix({"diff": f"{loss.item():.3f}"})
        _save(model, optim, 0, steps, args.out_dir, "last")
        return

    for epoch in range(start_epoch, args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            loss = do_step(batch)
            if global_step % args.log_interval == 0:
                pbar.set_postfix({"diff": f"{loss.item():.3f}"})
            global_step += 1
            if args.max_steps and global_step >= args.max_steps:
                pbar.close()
                print(f"[train_diffvc.py] reached --max-steps={args.max_steps}")
                _save(model, optim, epoch, global_step, args.out_dir, "last")
                val.final_eval(epochs_done, global_step)
                return
        epochs_done = epoch + 1
        _save(model, optim, epochs_done, global_step, args.out_dir, f"epoch{epochs_done}")
        val.dev_epoch(epochs_done, global_step)
    val.final_eval(epochs_done, global_step)


if __name__ == "__main__":
    main()
