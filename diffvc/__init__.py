"""DiffVC source-speech reconstructor.

Diffusion-based VC (vendored from https://github.com/huawei-noah/Speech-Backbones/tree/main/DiffVC,
MIT, see diffvc/LICENSE) adapted to condition on the external source-speaker embedding
`g`: a frozen pretrained LibriTTS "average voice" encoder supplies the content prior, and
a diffusion decoder (trained from scratch, `use_ref_t=False`) generates the source mel.
Mel front-end in `mel.py`, HiFi-GAN vocoder in `hifigan.py`/`vocoder.py`. Training entry
point is `train_diffvc.py`.
"""

from diffvc.reconstructor import DiffVCReconstructor, build_model

__all__ = ["DiffVCReconstructor", "build_model"]
