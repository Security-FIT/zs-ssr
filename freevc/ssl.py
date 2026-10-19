"""Pluggable self-supervised (SSL) content backend for the FreeVC reconstructor.

FreeVC extracts content features from a frozen SSL model, WavLM by default. The backend
can be swapped without touching the rest of the pipeline:

* ``wavlm``: Coqui's bundled WavLM-Large (no extra dependencies, as in upstream FreeVC).
* any HuggingFace id or local dir, loaded with `transformers` (`AutoModel`), e.g.
  ``facebook/wav2vec2-xls-r-300m``, ``microsoft/wavlm-large``, ``facebook/wav2vec2-large-xlsr-53``.

All backends return the content as ``(B, ssl_dim, T)`` at 50 Hz (16 kHz input, the same hop
of 320 as the spectrogram) and expose ``.ssl_dim`` for the model config. The backend is frozen
and kept out of the trained model's state_dict.

``layer=None`` uses the last hidden state, FreeVC's WavLM default (its ``get_content`` calls
``extract_features`` without ``output_layer``). kNN-VC uses layer 6 instead. XLS-R/wav2vec2
lack WavLM's speaker-disentanglement pretraining, so for those the layer matters more.
"""

from __future__ import annotations

import torch


class SSLBackend:
    """Frozen SSL feature extractor. `extract(wav)` -> (B, ssl_dim, T)."""

    ssl_dim: int
    sample_rate: int = 16000

    @torch.no_grad()
    def extract(self, wav: torch.Tensor) -> torch.Tensor:  # wav: (B, L) in [-1, 1]
        raise NotImplementedError

    def to(self, device):
        return self


class CoquiWavLMBackend(SSLBackend):
    """Coqui's bundled WavLM-Large (the upstream FreeVC content encoder)."""

    def __init__(self, device: str = "cpu", layer: int | None = None):
        from TTS.vc.layers.freevc.wavlm import get_wavlm

        self.model = get_wavlm(device)
        self.model.eval()
        self.device = device
        self.layer = layer
        self.ssl_dim = 1024

    @torch.no_grad()
    def extract(self, wav: torch.Tensor) -> torch.Tensor:
        c = self.model.extract_features(wav, output_layer=self.layer)[0]
        return c.transpose(1, 2)  # (B, 1024, T)

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self


class HFTransformersBackend(SSLBackend):
    """Any wav2vec2/WavLM-family checkpoint via HuggingFace `transformers`."""

    def __init__(self, model_id: str, device: str = "cpu", layer: int | None = None):
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.layer = layer
        self.ssl_dim = int(self.model.config.hidden_size)

    @torch.no_grad()
    def extract(self, wav: torch.Tensor) -> torch.Tensor:
        # wav2vec2/XLS-R/WavLM(HF) expect zero-mean/unit-var input (do_normalize=True).
        x = (wav - wav.mean(dim=1, keepdim=True)) / torch.sqrt(wav.var(dim=1, keepdim=True) + 1e-7)
        if self.layer is None:
            feats = self.model(x).last_hidden_state
        else:
            feats = self.model(x, output_hidden_states=True).hidden_states[self.layer]
        return feats.transpose(1, 2)  # (B, ssl_dim, T)

    def to(self, device):
        self.model.to(device)
        self.device = device
        return self


def build_ssl_backend(spec: str, layer: int | None = None, device: str = "cpu") -> SSLBackend:
    """`spec` is "wavlm" (Coqui bundled) or an HF model id / local dir (XLS-R, etc.)."""
    if spec == "wavlm":
        return CoquiWavLMBackend(device=device, layer=layer)
    return HFTransformersBackend(spec, device=device, layer=layer)
