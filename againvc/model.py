"""AGAIN-VC adapted to train a zero-shot source-speech reconstructor.

Vendored from AGAIN-VC (https://github.com/KimythAnly/AGAIN-VC, MIT, see againvc/LICENSE),
`model/again.py`. AGAIN-VC is an activation-guidance + adaptive instance normalization (AdaIN)
auto-encoder: the `Encoder` maps a mel to a content bottleneck (instance norm removes the
speaker information) and to per-block IN statistics, and the `Decoder` renders a mel from the
content, applying AdaIN with the speaker's IN statistics. Upstream takes those from a second
encoder pass over a reference mel.

Changes for reconstruction (each is commented at its site):
* The speaker AdaIN parameters are predicted from the external source-speaker embedding `g`
  by `SpeakerAdaptor` instead of a reference-mel encoder pass. The content still comes from
  `Encoder(VC-sample mel)` with the activation-guided IN bottleneck.
* Upstream's `train_step` / `inference_step` / `build_model` and the `util.mytorch` import are
  dropped (`train_againvc.py` has the training loop), as is the unused `ConvNorm2d`. The other
  modules are unchanged apart from the `Decoder.forward` signature.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================
#  Modules (vendored unchanged from AGAIN-VC model/again.py)
# ====================================
class InstanceNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def calc_mean_std(self, x, mask=None):
        B, C = x.shape[:2]
        mn = x.view(B, C, -1).mean(-1)
        sd = (x.view(B, C, -1).var(-1) + self.eps).sqrt()
        mn = mn.view(B, C, *((len(x.shape) - 2) * [1]))
        sd = sd.view(B, C, *((len(x.shape) - 2) * [1]))
        return mn, sd

    def forward(self, x, return_mean_std=False):
        mean, std = self.calc_mean_std(x)
        x = (x - mean) / std
        if return_mean_std:
            return x, mean, std
        else:
            return x


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, groups=1, bias=True, w_init_gain='linear', padding_mode='zeros', sn=False):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    groups=groups, bias=bias, padding_mode=padding_mode)
        if sn:
            self.conv = nn.utils.spectral_norm(self.conv)
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        return self.conv(signal)


class EncConvBlock(nn.Module):
    def __init__(self, c_in, c_h, subsample=1):
        super().__init__()
        self.seq = nn.Sequential(
            ConvNorm(c_in, c_h, kernel_size=3, stride=1),
            nn.BatchNorm1d(c_h),
            nn.LeakyReLU(),
            ConvNorm(c_h, c_in, kernel_size=3, stride=subsample),
        )
        self.subsample = subsample

    def forward(self, x):
        y = self.seq(x)
        if self.subsample > 1:
            x = F.avg_pool1d(x, kernel_size=self.subsample)
        return x + y


class DecConvBlock(nn.Module):
    def __init__(self, c_in, c_h, c_out, upsample=1):
        super().__init__()
        self.dec_block = nn.Sequential(
            ConvNorm(c_in, c_h, kernel_size=3, stride=1),
            nn.BatchNorm1d(c_h),
            nn.LeakyReLU(),
            ConvNorm(c_h, c_in, kernel_size=3),
        )
        self.gen_block = nn.Sequential(
            ConvNorm(c_in, c_h, kernel_size=3, stride=1),
            nn.BatchNorm1d(c_h),
            nn.LeakyReLU(),
            ConvNorm(c_h, c_in, kernel_size=3),
        )
        self.upsample = upsample

    def forward(self, x):
        y = self.dec_block(x)
        if self.upsample > 1:
            x = F.interpolate(x, scale_factor=self.upsample)
            y = F.interpolate(y, scale_factor=self.upsample)
        y = y + self.gen_block(y)
        return x + y


class Encoder(nn.Module):
    def __init__(self, c_in, c_out, n_conv_blocks, c_h, subsample):
        super().__init__()
        self.inorm = InstanceNorm()
        self.conv1d_first = ConvNorm(c_in * 1, c_h)
        self.conv1d_blocks = nn.ModuleList([
            EncConvBlock(c_h, c_h, subsample=sub) for _, sub in zip(range(n_conv_blocks), subsample)
        ])
        self.out_layer = ConvNorm(c_h, c_out)

    def forward(self, x):
        y = x
        y = y.squeeze(1)  # (B,1,80,T)->(B,80,T), a no-op when called with (B,80,T)
        y = self.conv1d_first(y)
        mns = []
        sds = []
        for block in self.conv1d_blocks:
            y = block(y)
            y, mn, sd = self.inorm(y, return_mean_std=True)
            mns.append(mn)
            sds.append(sd)
        y = self.out_layer(y)
        return y, mns, sds


class Decoder(nn.Module):
    def __init__(self, c_in, c_h, c_out, n_conv_blocks, upsample):
        super().__init__()
        self.in_layer = ConvNorm(c_in, c_h, kernel_size=3)
        self.act = nn.LeakyReLU()
        self.conv_blocks = nn.ModuleList([
            DecConvBlock(c_h, c_h, c_h, upsample=up) for _, up in zip(range(n_conv_blocks), upsample)
        ])
        self.inorm = InstanceNorm()
        self.rnn = nn.GRU(c_h, c_h, 2)
        self.out_layer = nn.Linear(c_h, c_out)

    def forward(self, content, spk_stats):
        """Render a mel from source `content` using `g`-derived AdaIN parameters.

        CHANGED vs upstream `Decoder.forward(enc, cond)`: upstream took `cond`, a second
        encoder pass over a reference mel, and read the speaker AdaIN parameters from it
        (`calc_mean_std(cond)` for the initial affine, `cond`'s per-block IN stats). Here they
        arrive as `spk_stats`, predicted from `g` by `SpeakerAdaptor`. The rest (content
        normalization, blocks, the prepended-frame GRU quirk, output projection) is unchanged.
        """
        mn0, sd0, mns, sds = spk_stats           # mn0/sd0: (B,c_in,1), mns/sds: [n_blocks x (B,c_h,1)]
        c = self.inorm(content)                  # normalize the content bottleneck
        c_affine = c * sd0 + mn0                  # initial AdaIN from g
        y = self.in_layer(c_affine)
        y = self.act(y)
        last_mn = mn0
        for block, mn, sd in zip(self.conv_blocks, mns, sds):
            y = block(y)
            y = self.inorm(y)
            y = y * sd + mn                       # per-block AdaIN from g
            last_mn = mn
        # upstream quirk preserved: prepend one frame (the last block's mn) before the
        # GRU and drop it afterwards.
        y = torch.cat((last_mn, y), dim=2)
        y = y.transpose(1, 2)
        y, _ = self.rnn(y)
        y = y[:, 1:, :]
        y = self.out_layer(y)
        y = y.transpose(1, 2)
        return y


class VariantSigmoid(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return 1 / (1 + torch.exp(-self.alpha * x))


class NoneAct(nn.Module):
    def forward(self, x):
        return x


class Activation(nn.Module):
    dct = {'none': NoneAct, 'sigmoid': VariantSigmoid, 'tanh': nn.Tanh}

    def __init__(self, act, params=None):
        super().__init__()
        self.act = Activation.dct[act](**(params or {}))

    def forward(self, x):
        return self.act(x)


# ====================================
#  Speaker conditioning from the external embedding g (NEW)
# ====================================
class SpeakerAdaptor(nn.Module):
    """Map the external source-speaker embedding `g` -> the decoder's AdaIN params.

    NEW for the reconstructor, replacing upstream's reference-mel encoder pass. A linear
    projection of `g` gives the initial affine `(mn0, sd0)` (size = decoder `c_in`, the content
    bottleneck) and per-block `(mn_i, sd_i)` (size = decoder `c_h`) for `n_blocks` blocks. The
    stds go through softplus because upstream's are `sqrt(var)` from InstanceNorm, i.e. > 0.
    """

    def __init__(self, emb_dim: int, c_in: int, c_h: int, n_blocks: int):
        super().__init__()
        self.c_in = c_in
        self.c_h = c_h
        self.n_blocks = n_blocks
        self.proj = nn.Linear(emb_dim, 2 * c_in + 2 * c_h * n_blocks)

    def forward(self, g):
        v = self.proj(g)
        i = 0
        mn0 = v[:, i:i + self.c_in]; i += self.c_in
        sd0 = v[:, i:i + self.c_in]; i += self.c_in
        mns, sds = [], []
        for _ in range(self.n_blocks):
            mns.append(v[:, i:i + self.c_h]); i += self.c_h
            sds.append(v[:, i:i + self.c_h]); i += self.c_h
        mn0 = mn0.unsqueeze(-1)
        sd0 = F.softplus(sd0).unsqueeze(-1) + 1e-4
        mns = [m.unsqueeze(-1) for m in mns]
        sds = [F.softplus(s).unsqueeze(-1) + 1e-4 for s in sds]
        return mn0, sd0, mns, sds


# default architecture from AGAIN-VC config/train_again-c4s.yaml
ENCODER_PARAMS = dict(c_in=80, c_h=256, c_out=4, n_conv_blocks=6, subsample=[1, 1, 1, 1, 1, 1])
DECODER_PARAMS = dict(c_in=4, c_h=256, c_out=80, n_conv_blocks=6, upsample=[1, 1, 1, 1, 1, 1])
ACTIVATION_PARAMS = dict(act='sigmoid', params={'alpha': 0.1})


class AgainReconstructor(nn.Module):
    """AGAIN-VC reconstructor: content = Encoder(VC mel), speaker = external `g`."""

    def __init__(self, emb_dim: int, encoder_params=None, decoder_params=None, activation_params=None):
        super().__init__()
        encoder_params = encoder_params or ENCODER_PARAMS
        decoder_params = decoder_params or DECODER_PARAMS
        activation_params = activation_params or ACTIVATION_PARAMS
        self.encoder = Encoder(**encoder_params)
        self.decoder = Decoder(**decoder_params)
        self.act = Activation(**activation_params)
        self.adaptor = SpeakerAdaptor(
            emb_dim, decoder_params['c_in'], decoder_params['c_h'], decoder_params['n_conv_blocks']
        )

    def forward(self, content_mel, g):
        """content_mel: (B, n_mels, T), g: (B, emb_dim) -> mel (B, n_mels, T)."""
        y, _, _ = self.encoder(content_mel)
        y = self.act(y)                       # activation guidance on the content bottleneck
        return self.decoder(y, self.adaptor(g))

    def inference(self, content_mel, g):
        """Full-length inference: pad time to a multiple of 8 (upstream parity), then crop."""
        T = content_mel.size(-1)
        pad = (8 - T % 8) % 8
        if pad:
            content_mel = F.pad(content_mel, (0, pad), mode='reflect')
        out = self.forward(content_mel, g)
        return out[..., :T]


def main():
    """Self-test: shapes round-trip through the reconstructor."""
    torch.manual_seed(0)
    m = AgainReconstructor(emb_dim=128)
    content = torch.randn(2, 80, 128)
    g = torch.randn(2, 128)
    out = m(content, g)
    assert out.shape == (2, 80, 128), out.shape
    inf = m.inference(content[:1], g[:1])
    assert inf.shape == (1, 80, 128), inf.shape
    n = sum(p.numel() for p in m.parameters())
    print(f"[model.py] forward OK: content{tuple(content.shape)} + g{tuple(g.shape)} "
          f"-> {tuple(out.shape)}; params={n/1e6:.2f}M")


if __name__ == "__main__":
    main()
