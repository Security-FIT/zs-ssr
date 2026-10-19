"""AGAIN-VC source-speech reconstructor.

AdaIN auto-encoder (vendored from https://github.com/KimythAnly/AGAIN-VC, MIT,
see againvc/LICENSE) adapted to condition on the external source-speaker embedding
`g` instead of a reference utterance. Mel front-end in `dsp.py`, MelGAN vocoder in
`vocoder.py`. The training entry point is `train_againvc.py`.
"""

from againvc.model import AgainReconstructor

__all__ = ["AgainReconstructor"]
