"""Train a zero-shot source-speech reconstructor (FreeVC).

Input = the VC sample's WavLM features (content), speaker = the source-speaker embedding `g`,
target = the original LibriSpeech utterance (spectrogram and waveform). The model is the FreeVC
of Coqui TTS (https://github.com/idiap/coqui-ai-TTS, MPL-2.0, installed as the `coqui-tts`
package, not vendored) with the external `g` (`FreeVCReconstructor`). The loop follows upstream
FreeVC's `train.py` (https://github.com/OlaWod/FreeVC, MIT, see freevc/LICENSE): a multi-period
discriminator, then the generator with mel, KL, feature-matching and adversarial losses.

Single GPU. Run it as a cluster job (scripts/sge/freevc/). Spectrogram and
WavLM features are computed on the GPU.

Example:
    python train_freevc.py \
        --exp-npz embeddings/non-finetuned/organizer-pretrained/exported_embeddings/train_embeddings.npz \
        --dev-npz .../dev_embeddings.npz --test-npz .../test_embeddings.npz \
        --train-synth Train-3 --dev-synth Dev-3 --test-synth Test-3 \
        --slt-root ../datasets/slt-sstc --librispeech-root ../datasets/LibriSpeech \
        --out-dir outputs/freevc/synth-3 --batch-size 16 --epochs 12 --wavlm-cache wavlm_cache
"""

from __future__ import annotations

import argparse
import functools
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from freevc import (
    FreeVCReconstructor,
    MultiPeriodDiscriminator,
    ParallelReconDataset,
    collate,
)
from freevc.audio import mel_spectrogram_torch, spec_to_mel_torch, spectrogram_torch
from freevc.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from freevc.ssl import build_ssl_backend
from freevc.validation import reconstruct as freevc_reconstruct
from freevc.wavlm_cache import generate_subset

from dataset.npz_cache import NpzCache, subset_file_paths
from dataset.pairing import build_parallel_samples
from dataset.sltsstc_tar import activate as activate_corpus
from dataset.sltsstc_tar import read_wav
from train_common import Validator, add_common_args, smoke_limit

from TTS.vc.configs.freevc_config import FreeVCArgs, FreeVCAudioConfig, FreeVCConfig
from TTS.vc.layers.freevc.commons import clip_grad_value_, slice_segments


def build_model(ssl_dim: int, emb_dim: int, device: str, ssl_backend=None):
    """FreeVCReconstructor. ssl_dim = content-feature dim, gin_channels from the embedding.

    `ssl_backend` is None in cache mode (WavLM not loaded) and `ssl_dim` then comes from
    the cached feature shape.
    """
    config = FreeVCConfig(
        model_args=FreeVCArgs(
            gin_channels=emb_dim,            # = source-embedding dim (e.g. 256), train from scratch
            ssl_dim=ssl_dim,                 # content-feature dim (WavLM 1024, XLS-R-300m 1024, 1B 1280, ...)
            use_spk=True,                    # external source embedding as g
            segment_size=8960,               # samples, -> frames internally
        ),
        audio=FreeVCAudioConfig(input_sample_rate=16000, output_sample_rate=16000),
    )
    model = FreeVCReconstructor(config, ssl_backend=ssl_backend).to(device)
    return model, config


def _save(model, net_d, optim_g, optim_d, epoch, step, out_dir, tag):
    """Checkpoint as freevc_ssr_<tag>.pth, tag = epoch<E> or last.

    Written via tmp+rename so a kill can't truncate an existing checkpoint.
    """
    path = os.path.join(out_dir, f"freevc_ssr_{tag}.pth")
    torch.save({"model": model.state_dict(), "net_d": net_d.state_dict(),
                "optim_g": optim_g.state_dict(), "optim_d": optim_d.state_dict(),
                "epoch": epoch, "step": step}, path + ".tmp")
    os.replace(path + ".tmp", path)
    print(f"[train_freevc.py] saved {path}")


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser, batch_size=16, align_max_lag=64)  # FreeVC frames are 20 ms
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--c-mel", type=float, default=45.0)
    parser.add_argument("--c-kl", type=float, default=1.0)
    parser.add_argument("--ssl-model", default="wavlm",
                        help='content SSL backend: "wavlm" (Coqui bundled) or an HF id / local dir '
                             "(e.g. facebook/wav2vec2-xls-r-300m, microsoft/wavlm-large)")
    parser.add_argument("--ssl-layer", type=int, default=None,
                        help="SSL hidden layer for the content; default the last layer, as in FreeVC")
    parser.add_argument("--wavlm-cache", default=None,
                        help="dir of per-subset WavLM features <cache>/<subset>.npz, read instead of running "
                             "WavLM every step; missing subsets (train/dev/test) are generated on startup")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "GPU required (run it as a cluster job)."
    if args.align and args.wavlm_cache:
        raise SystemExit("[train_freevc.py] --align needs WavLM to run live on the shifted audio; "
                         "it can't be combined with --wavlm-cache")
    device = "cuda"
    os.makedirs(args.out_dir, exist_ok=True)
    activate_corpus(args.slt_root)  # tar-packed slt-root? read VC wavs from the tars, else loose

    wavlm_cache = None
    if args.wavlm_cache:
        wavlm_cache = NpzCache(args.wavlm_cache, args.slt_root)
        need = subset_file_paths(args.exp_npz, args.train_synth)
        if args.dev_npz:
            need.update(subset_file_paths(args.dev_npz, args.dev_synth))
        if args.test_npz:
            need.update(subset_file_paths(args.test_npz, args.test_synth))
        missing = {subset: paths for subset, paths in need.items() if not wavlm_cache.has(subset)}
        if missing:
            print(f"[train_freevc.py] {sorted(missing)} not in the WavLM cache, generating "
                  f"({sum(len(paths) for paths in missing.values())} utts)")
            gen_ssl = build_ssl_backend(args.ssl_model, layer=args.ssl_layer, device=device)
            load_wav = lambda path: read_wav(os.path.join(args.slt_root, path), 16000)
            for subset, paths in missing.items():
                generate_subset(args.wavlm_cache, subset, paths, load_wav, gen_ssl, device)
            del gen_ssl  # free WavLM before training
            torch.cuda.empty_cache()

    samples = build_parallel_samples(
        args.exp_npz, args.slt_root, args.librispeech_root,
        average=args.average_emb, normalize=args.normalize_emb, limit=smoke_limit(args),
        synthesizers=args.train_synth,
    )
    assert samples, "no (VC, LibriSpeech) pairs found (check --slt-root / --librispeech-root vs --exp-npz)"
    emb_dim = samples[0][2].shape[-1]
    print(f"[train_freevc.py] {len(samples)} (VC->LibriSpeech) pairs, g dim={emb_dim}, "
          f"normalize_emb={args.normalize_emb}, average_emb={args.average_emb}")

    if wavlm_cache is not None:  # WavLM is never called, so don't load it
        ssl_backend = None
        ssl_dim = wavlm_cache.feature_dim(next(iter(need)))
        print(f"[train_freevc.py] WavLM cache: ssl_dim={ssl_dim}, WavLM not loaded")
    else:
        ssl_backend = build_ssl_backend(args.ssl_model, layer=args.ssl_layer, device=device)
        ssl_dim = ssl_backend.ssl_dim
        print(f"[train_freevc.py] SSL backend '{args.ssl_model}' (layer={args.ssl_layer}) ssl_dim={ssl_dim}")

    model, config = build_model(ssl_dim, emb_dim, device, ssl_backend=ssl_backend)
    net_d = MultiPeriodDiscriminator(config.model_args.use_spectral_norm).to(device)
    optim_g = torch.optim.AdamW(model.parameters(), args.lr, betas=(0.8, 0.99), eps=1e-9)
    optim_d = torch.optim.AdamW(net_d.parameters(), args.lr, betas=(0.8, 0.99), eps=1e-9)

    audio_cfg = config.audio
    n_fft, hop, win = audio_cfg.filter_length, audio_cfg.hop_length, audio_cfg.win_length
    n_mel, sr = audio_cfg.n_mel_channels, audio_cfg.input_sample_rate
    fmin, fmax = audio_cfg.mel_fmin, audio_cfg.mel_fmax
    seg_f = model.segment_size_frames

    # --align can crop a pair below one decoder segment, which makes the random segment slice
    # empty, pad to a segment plus 2 frames of WavLM/STFT rounding
    ds_kwargs = {"min_wav_len": (seg_f + 2) * hop} if args.align else {}
    train_ds = ParallelReconDataset(samples, align=args.align, align_max_lag=args.align_max_lag,
                                    wavlm_cache=wavlm_cache, **ds_kwargs)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True, collate_fn=collate,
    )

    start_epoch, global_step = 0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        net_d.load_state_dict(ckpt["net_d"])
        optim_g.load_state_dict(ckpt["optim_g"])
        optim_d.load_state_dict(ckpt["optim_d"])
        start_epoch, global_step = ckpt["epoch"], ckpt["step"]
        print(f"[train_freevc.py] resumed from {args.resume} @ epoch {start_epoch} step {global_step}")
    epochs_done = start_epoch  # completed epochs, used to name the validation results

    # with the cache, validation also reads the WavLM features from it
    recon_fn = (functools.partial(freevc_reconstruct, wavlm_cache=wavlm_cache)
                if wavlm_cache is not None else freevc_reconstruct)
    val = Validator(args, model, device, recon_fn, meta={"architecture": "freevc"})

    def do_step(batch):
        """One discriminator and one generator step, returns the loss tensors for logging."""
        content, tgt, lengths, g = (tensor.to(device, non_blocking=True) for tensor in batch)

        # spectrogram, mel and waveform targets come from the LibriSpeech original, the content
        # from the VC sample (same utterance, so frame-aligned)
        spec = spectrogram_torch(tgt, n_fft, sr, hop, win)            # (B, n_fft//2+1, T)
        if content.dim() == 3:
            content_feats = content                                  # cached WavLM features (B, ssl_dim, Tc)
        else:
            with torch.no_grad():
                content_feats = model.extract_wavlm_features(content)  # WavLM run live on the VC wav
        n_frames = min(spec.size(-1), content_feats.size(-1))
        spec, content_feats = spec[..., :n_frames], content_feats[..., :n_frames]
        spec_lengths = (lengths // hop).clamp(max=n_frames)
        mel = spec_to_mel_torch(spec, n_fft, n_mel, sr, fmin, fmax)   # (B, n_mel, T)

        # VITS outputs: the generated waveform, the frame indices of the random segment
        # the decoder was run on, the frame mask, and the prior/posterior distributions
        # the KL term compares (z_post_flowed = flow(z_post), mapped into the prior space)
        (wav_gen, seg_start_frames, frame_mask,
         (z_post, z_post_flowed, m_prior, logs_prior, m_post, logs_post)) = model(
            content_feats, spec, g, spec_lengths=spec_lengths
        )
        mel_real_seg = slice_segments(mel, seg_start_frames, seg_f)
        mel_gen_seg = mel_spectrogram_torch(wav_gen.squeeze(1), n_fft, n_mel, sr, hop, win, fmin, fmax)
        # the matching target-waveform segment, (B, 1, seg_samples)
        wav_real_seg = slice_segments(tgt.unsqueeze(1), seg_start_frames * hop, seg_f * hop)

        disc_real, disc_gen, _, _ = net_d(wav_real_seg, wav_gen.detach())
        loss_disc, _, _ = discriminator_loss(disc_real, disc_gen)
        optim_d.zero_grad()
        loss_disc.backward()
        clip_grad_value_(net_d.parameters(), None)
        optim_d.step()

        disc_real, disc_gen, fmap_real, fmap_gen = net_d(wav_real_seg, wav_gen)
        loss_mel = F.l1_loss(mel_real_seg, mel_gen_seg) * args.c_mel
        # kl_loss is positional: (z_p, logs_q, m_p, logs_p, z_mask)
        loss_kl = kl_loss(z_post_flowed, logs_post, m_prior, logs_prior, frame_mask) * args.c_kl
        loss_fm = feature_loss(fmap_real, fmap_gen)
        loss_gen, _ = generator_loss(disc_gen)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
        optim_g.zero_grad()
        loss_gen_all.backward()
        clip_grad_value_(model.parameters(), None)
        optim_g.step()
        return loss_disc, loss_gen_all, loss_mel, loss_kl, loss_fm

    def postfix(losses):
        """tqdm postfix: do_step returns (disc, gen_all, mel, kl, feature-match)."""
        disc, gen_all, mel, kl, fm = (loss.item() for loss in losses)
        return {"g": f"{gen_all:.2f}", "d": f"{disc:.2f}", "mel": f"{mel:.2f}",
                "kl": f"{kl:.2f}", "fm": f"{fm:.2f}"}

    model.train()
    net_d.train()

    if args.overfit_one_batch:
        steps = args.max_steps or 200
        batch = next(iter(train_loader))
        print(f"[train_freevc.py] overfit-one-batch: {steps} steps on one batch (B={batch[0].size(0)})")
        pbar = tqdm(range(steps), desc="overfit")
        for step in pbar:
            losses = do_step(batch)
            if step % args.log_interval == 0 or step == steps - 1:
                pbar.set_postfix(postfix(losses))
        _save(model, net_d, optim_g, optim_d, 0, steps, args.out_dir, "last")
        return

    for epoch in range(start_epoch, args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        for batch in pbar:
            losses = do_step(batch)
            if global_step % args.log_interval == 0:
                pbar.set_postfix(postfix(losses))
            global_step += 1
            if args.max_steps and global_step >= args.max_steps:
                pbar.close()
                print(f"[train_freevc.py] reached --max-steps={args.max_steps}")
                _save(model, net_d, optim_g, optim_d, epoch, global_step, args.out_dir, "last")
                val.final_eval(epochs_done, global_step)
                return
        epochs_done = epoch + 1
        _save(model, net_d, optim_g, optim_d, epochs_done, global_step, args.out_dir, f"epoch{epochs_done}")
        val.dev_epoch(epochs_done, global_step)
    val.final_eval(epochs_done, global_step)


if __name__ == "__main__":
    main()
