"""DiffVC adapted to a zero-shot source-speech reconstructor.

`DiffVCReconstructor` subclasses the vendored `DiffVC` (diffvc/model/vc.py) and wires
it for parallel reconstruction with an external source-speaker embedding `g`:

* Encoder = DiffVC's pretrained LibriTTS "average voice" encoder, frozen (loaded from
  `enc.pt`). It normalizes away the speaker, so on a VC-sample mel it gives that utterance's
  content in the average voice whatever the converted target voice was. It plays the role
  WavLM has in FreeVC.
* Decoder = the diffusion score model, trained from scratch, conditioned on the
  external `g` (`use_ref_t=False`, so no reference mel is needed at inference).
* Training target `x0` = the LibriSpeech original mel. Content prior `mean` =
  encoder(VC-sample mel) (or a cached avg-voice mel, see `--libritts-cache`).

Only the wiring is new: the diffusion math is the vendored `Diffusion`/`FwdDiffusion`.
"""

from __future__ import annotations

import torch

from diffvc.model.utils import fix_len_compatibility, sequence_mask
from diffvc.model.vc import DiffVC

# DiffVC architecture constants (from DiffVC params.py). use_ref_t=False (g-only) and
# emb_dim (external g dim) are supplied per build.
DIFFVC_ARCH = dict(n_feats=80, channels=192, filters=768, heads=2, layers=6, kernel=3,
                   dropout=0.1, window_size=4, enc_dim=128, spk_dim=128, dec_dim=256,
                   beta_min=0.05, beta_max=20.0)


class DiffVCReconstructor(DiffVC):
    def train(self, mode: bool = True):
        # Keep the frozen "average voice" encoder in eval mode even when the model is in
        # train mode: its FwdDiffusion.forward runs under no_grad, but dropout would still
        # be active in train mode and perturb the (frozen) content prior.
        super().train(mode)
        self.encoder.eval()
        return self

    def compute_loss(self, target, lengths, c, content=None, mean=None):
        """Diffusion score-matching loss for reconstruction.

        CHANGED vs upstream `DiffVC.compute_loss`: the diffusion prior `mean` comes from the
        VC sample, computed from `content` by the frozen encoder or passed in precomputed (the
        libritts cache), while the target `x0` is the LibriSpeech original `target`. With
        `use_ref_t=False` the reference arguments of the decoder loss (`target`, `mean`) are
        dummies the score estimator ignores, and `c` is the external source embedding `g`.
        """
        mask = sequence_mask(lengths).unsqueeze(1).to(target.dtype)
        if mean is None:
            assert content is not None, "compute_loss needs `content` or a precomputed `mean`"
            mean = self.encoder(content, mask).detach()
        return self.decoder.compute_loss(target, mask, mean, target, mean, c)

    @torch.no_grad()
    def reconstruct(self, content, lengths, c, n_timesteps: int = 6, mode: str = "ml"):
        """VC content mel + external `g` -> generated (LibriSpeech-domain) mel.

        Mirrors the vendored `DiffVC.forward` but drives the prior from `content` and drops
        the reference path (use_ref_t=False): a dummy reference (`mean_new`) is passed to the
        decoder and ignored by the estimator.
        """
        content, lengths, c = self.relocate_input([content, lengths, c])
        x_mask = sequence_mask(lengths).unsqueeze(1).to(content.dtype)
        mean = self.encoder(content, x_mask)
        mean_x = self.decoder.compute_diffused_mean(content, x_mask, mean, 1.0)

        b = content.shape[0]
        max_length = int(lengths.max())
        max_length_new = fix_len_compatibility(max_length)
        x_mask_new = sequence_mask(lengths, max_length_new).unsqueeze(1).to(content.dtype)
        mean_new = torch.zeros((b, self.n_feats, max_length_new), dtype=content.dtype, device=content.device)
        mean_x_new = torch.zeros((b, self.n_feats, max_length_new), dtype=content.dtype, device=content.device)
        for i in range(b):
            mean_new[i, :, :lengths[i]] = mean[i, :, :lengths[i]]
            mean_x_new[i, :, :lengths[i]] = mean_x[i, :, :lengths[i]]

        z = mean_x_new + torch.randn_like(mean_x_new, device=mean_x_new.device)
        y = self.decoder(z, x_mask_new, mean_new, mean_new, x_mask_new, mean_new, c, n_timesteps, mode)
        return y[:, :, :max_length]


def build_model(emb_dim: int, device: str, enc_ckpt: str) -> DiffVCReconstructor:
    """Frozen pretrained encoder + trainable g-conditioned diffusion decoder."""
    model = DiffVCReconstructor(**DIFFVC_ARCH, use_ref_t=False, emb_dim=emb_dim).to(device)
    model.load_encoder(enc_ckpt)
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    model.encoder.eval()
    return model


def main():
    """Self-test on CPU with random weights (no enc.pt): the loss and a short sampling pass run
    and the shapes round-trip. T must be a multiple of 4 (two U-Net downsamplings)."""
    torch.manual_seed(0)
    emb_dim, B, T = 128, 2, 64
    model = DiffVCReconstructor(**DIFFVC_ARCH, use_ref_t=False, emb_dim=emb_dim)
    model.train()  # exercises the encoder-stays-eval override
    content = torch.randn(B, 80, T)
    target = torch.randn(B, 80, T)
    lengths = torch.LongTensor([T, T])
    g = torch.randn(B, emb_dim)
    loss = model.compute_loss(target, lengths, g, content=content)
    assert loss.ndim == 0 and torch.isfinite(loss), loss
    out = model.reconstruct(content[:1], lengths[:1], g[:1], n_timesteps=2)
    assert out.shape[0] == 1 and out.shape[1] == 80, out.shape
    n = sum(p.numel() for p in model.decoder.parameters())
    print(f"[reconstructor.py] loss={loss.item():.3f}, reconstruct->{tuple(out.shape)}, "
          f"decoder params={n/1e6:.2f}M; encoder frozen in train mode: {not model.encoder.training}")


if __name__ == "__main__":
    main()
