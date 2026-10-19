"""FreeVC source-speech reconstructor.

Coqui's FreeVC (VITS + HiFi-GAN, WavLM content) conditioned on the external source-speaker
embedding `g`. Model in `model.py`, data in `dataset.py`, content backends in `ssl.py`, the WavLM
cache in `wavlm_cache.py`. Training entry point is `train_freevc.py`.
"""

from .dataset import ParallelReconDataset, collate
from .model import FreeVCReconstructor, MultiPeriodDiscriminator
from .ssl import build_ssl_backend

__all__ = [
    "FreeVCReconstructor",
    "MultiPeriodDiscriminator",
    "ParallelReconDataset",
    "collate",
    "build_ssl_backend",
]
