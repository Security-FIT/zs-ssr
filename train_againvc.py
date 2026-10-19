"""Train a zero-shot source-speech reconstructor (AGAIN-VC).

Input = the VC sample's AGAIN log10 mel (content), speaker = the source-speaker embedding `g`,
target = the mel of the original LibriSpeech utterance. The model is AGAIN-VC's AdaIN
auto-encoder with the AdaIN speaker parameters predicted from `g` (`againvc/model.py`), trained
like AGAIN-VC: L1(reconstructed mel, target mel), one optimizer, no GAN. Based on AGAIN-VC
(https://github.com/KimythAnly/AGAIN-VC, MIT, see againvc/LICENSE). Validation turns the mel into
a waveform with the MelGAN vocoder of descriptinc/melgan-neurips, loaded via torch.hub
(`againvc/vocoder.py`).

Single GPU. Run it as a cluster job (scripts/sge/againvc/).

Example:
    python train_againvc.py \
        --exp-npz embeddings/non-finetuned/organizer-pretrained/exported_embeddings/train_embeddings.npz \
        --dev-npz .../dev_embeddings.npz --test-npz .../test_embeddings.npz \
        --train-synth Train-3 --dev-synth Dev-3 --test-synth Test-3 \
        --slt-root ../datasets/slt-sstc --librispeech-root ../datasets/LibriSpeech \
        --out-dir outputs/againvc/synth-3 --batch-size 32 --max-steps 100000
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from againvc.dataset import ParallelMelReconDataset, collate
from againvc.model import AgainReconstructor
from againvc.reconstruct import reconstruct as againvc_reconstruct
from dataset.pairing import build_parallel_samples
from dataset.sltsstc_tar import activate as activate_corpus
from train_common import Validator, add_common_args, smoke_limit


def _save(model, optim, epoch, step, out_dir, tag):
    """Checkpoint as againvc_ssr_<tag>.pth, tag = epoch<E> or last.

    Written via tmp+rename so a kill can't truncate an existing checkpoint.
    """
    path = os.path.join(out_dir, f"againvc_ssr_{tag}.pth")
    torch.save({"model": model.state_dict(), "optim": optim.state_dict(),
                "epoch": epoch, "step": step}, path + ".tmp")
    os.replace(path + ".tmp", path)
    print(f"[train_againvc.py] saved {path}")


def main():
    p = argparse.ArgumentParser()
    add_common_args(p, batch_size=32, align_max_lag=50)
    p.add_argument("--lr", type=float, default=5e-4, help="AGAIN-VC default")
    p.add_argument("--weight-decay", type=float, default=1e-4, help="AGAIN-VC default")
    p.add_argument("--grad-norm", type=float, default=3.0, help="AGAIN-VC default grad clip")
    p.add_argument("--seglen", type=int, default=128, help="training mel-frame crop (AGAIN-VC default)")
    args = p.parse_args()

    assert torch.cuda.is_available(), "GPU required (run it as a cluster job)."
    device = "cuda"
    os.makedirs(args.out_dir, exist_ok=True)
    activate_corpus(args.slt_root)  # tar-packed slt-root? read VC wavs from the tars, else loose

    samples = build_parallel_samples(
        args.exp_npz, args.slt_root, args.librispeech_root,
        average=args.average_emb, normalize=args.normalize_emb, limit=smoke_limit(args),
        synthesizers=args.train_synth,
    )
    assert samples, "no (VC, LibriSpeech) pairs found (check --slt-root / --librispeech-root vs --exp-npz)"
    emb_dim = samples[0][2].shape[-1]
    print(f"[train_againvc.py] {len(samples)} (VC->LibriSpeech) pairs, g dim={emb_dim}, "
          f"normalize_emb={args.normalize_emb}, average_emb={args.average_emb}")
    train_ds = ParallelMelReconDataset(samples, seglen=args.seglen,
                                       align=args.align, align_max_lag=args.align_max_lag)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True, collate_fn=collate,
    )

    model = AgainReconstructor(emb_dim=emb_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                             amsgrad=True, weight_decay=args.weight_decay)

    start_epoch, global_step = 0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        start_epoch, global_step = ckpt["epoch"], ckpt["step"]
        print(f"[train_againvc.py] resumed from {args.resume} @ epoch {start_epoch} step {global_step}")
    epochs_done = start_epoch  # completed epochs, used to name the validation results

    val = Validator(args, model, device, againvc_reconstruct, meta={"architecture": "againvc"})

    def do_step(batch):
        content, target, g = (t.to(device, non_blocking=True) for t in batch)
        out = model(content, g)
        loss = F.l1_loss(out, target)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
        optim.step()
        return loss

    model.train()

    if args.overfit_one_batch:
        steps = args.max_steps or 200
        batch = next(iter(train_loader))
        print(f"[train_againvc.py] overfit-one-batch: {steps} steps on one batch (B={batch[0].size(0)})")
        pbar = tqdm(range(steps), desc="overfit")
        for step in pbar:
            loss = do_step(batch)
            if step % args.log_interval == 0 or step == steps - 1:
                pbar.set_postfix({"l1": f"{loss.item():.3f}"})
        _save(model, optim, 0, steps, args.out_dir, "last")
        return

    for epoch in range(start_epoch, args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            loss = do_step(batch)
            if global_step % args.log_interval == 0:
                pbar.set_postfix({"l1": f"{loss.item():.3f}"})
            global_step += 1
            if args.max_steps and global_step >= args.max_steps:
                pbar.close()
                print(f"[train_againvc.py] reached --max-steps={args.max_steps}")
                _save(model, optim, epoch, global_step, args.out_dir, "last")
                val.final_eval(epochs_done, global_step)
                return
        epochs_done = epoch + 1
        _save(model, optim, epochs_done, global_step, args.out_dir, f"epoch{epochs_done}")
        val.dev_epoch(epochs_done, global_step)
    val.final_eval(epochs_done, global_step)


if __name__ == "__main__":
    main()
