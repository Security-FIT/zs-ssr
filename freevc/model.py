"""FreeVC adapted to train a zero-shot source-speech reconstructor.

Coqui's `TTS.vc.models.freevc.FreeVC` has the full architecture but is inference-only:
`train_step`/`eval_step` are empty stubs and `forward` sets `g = None`, ignoring any external
speaker embedding. This subclass makes it trainable:

* `forward` passes the external source-speaker embedding `g` to the posterior encoder (enc_q),
  the flow and the decoder, which is what `use_spk=True` is meant to do.
* `segment_size` is converted to frames for the decoder slice (Coqui stores it in samples but
  passes it to `rand_slice_segments`, which works in frames).
* the pretrained speaker encoder and Coqui's WavLM download are skipped: `g` and the content
  features come from outside.

Build it with `use_spk=True` and `gin_channels` = the embedding size (256 for the MFA-Conformer
embeddings, which `train_freevc.py` reads from the data). `MultiPeriodDiscriminator` is Coqui's,
re-exported as is.
"""

import torch
from torch import nn

import TTS.vc.models.freevc as _freevc_mod
from TTS.vc.layers.freevc.commons import rand_slice_segments
from TTS.vc.models.freevc import FreeVC, MultiPeriodDiscriminator  # noqa: F401 (re-exported)


class FreeVCReconstructor(FreeVC):
    def __init__(self, config, speaker_manager=None, ssl_backend=None):
        # skip Coqui's WavLM-Large download: the content comes from `ssl_backend`, or from the
        # WavLM cache, in which case `ssl_backend` is None and never loaded
        _orig_get_wavlm = _freevc_mod.get_wavlm
        _freevc_mod.get_wavlm = lambda *a, **k: nn.Identity()
        try:
            super().__init__(config, speaker_manager)
        finally:
            _freevc_mod.get_wavlm = _orig_get_wavlm

        if not self.use_spk:
            raise ValueError(
                "FreeVCReconstructor expects use_spk=True (external source embedding as g). "
                "Set config.model_args.use_spk=True and gin_channels to the embedding dim."
            )
        if ssl_backend is not None and ssl_backend.ssl_dim != self.args.ssl_dim:
            raise ValueError(
                f"ssl_backend dim {ssl_backend.ssl_dim} != config ssl_dim {self.args.ssl_dim}; "
                "set FreeVCArgs.ssl_dim to the backend's hidden size."
            )
        # in a tuple so the frozen SSL model is not a submodule (stays out of the checkpoint
        # and the optimizer)
        self._ssl_holder = (ssl_backend,)
        # args.segment_size is in samples, the decoder slice works in frames
        self.segment_size_frames = self.args.segment_size // config.audio.hop_length

    @property
    def ssl_backend(self):
        return self._ssl_holder[0]

    def extract_wavlm_features(self, y):
        """Content features via the pluggable SSL backend. Returns (B, ssl_dim, T)."""
        if self.ssl_backend is None:
            raise RuntimeError(
                "[model.py] no SSL backend loaded (--wavlm-cache mode) and this utterance is not "
                "in the cache; cache it or run without --wavlm-cache"
            )
        return self.ssl_backend.extract(y)

    def load_pretrained_speaker_encoder(self):
        """No-op: we supply external source-speaker embeddings as `g`."""
        return

    def forward(self, c, spec, g, c_lengths=None, spec_lengths=None):
        """Training forward.

        Args:
            c:    WavLM content features, (B, ssl_dim, T).
            spec: linear spectrogram, (B, spec_channels, T).
            g:    source-speaker embedding, (B, gin_channels) or (B, gin_channels, 1).
            c_lengths, spec_lengths: frame lengths, (B,).
        Returns:
            (o, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q))
        """
        if g is None:
            raise ValueError("FreeVCReconstructor.forward requires external g (source embedding).")
        if g.dim() == 2:
            g = g.unsqueeze(-1)  # (B, gin_channels, 1)

        if c_lengths is None:
            c_lengths = (torch.ones(c.size(0), device=c.device) * c.size(-1)).long()
        if spec_lengths is None:
            spec_lengths = (torch.ones(spec.size(0), device=spec.device) * spec.size(-1)).long()

        _, m_p, logs_p, _ = self.enc_p(c, c_lengths)
        z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)
        z_p = self.flow(z, spec_mask, g=g)

        z_slice, ids_slice = rand_slice_segments(z, spec_lengths, self.segment_size_frames)
        o = self.dec(z_slice, g=g)
        return o, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q)
