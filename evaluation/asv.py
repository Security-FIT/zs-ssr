"""Speaker verification (ASV) for scoring reconstructions: an ECAPA-TDNN wrapper and the EER.

Every reconstructor is scored the same way: the reconstruction and a real utterance of the
source speaker are embedded with the WeSpeaker ECAPA-TDNN and compared by cosine. The model
directory is `evaluation/models/wespeaker/ecapa-tdnn1024_lm`. `wespeaker` is imported lazily,
so training without in-loop validation does not need it.
"""

from __future__ import annotations

import os

import librosa
import numpy as np
import torch


def compute_eer(target_scores, nontarget_scores) -> float:
    """Equal error rate of target / non-target similarity scores."""
    target = np.asarray(target_scores, dtype=np.float64)
    nontarget = np.asarray(nontarget_scores, dtype=np.float64)
    if len(target) == 0 or len(nontarget) == 0:
        return float("nan")
    target.sort()
    nontarget.sort()
    thr = np.sort(np.unique(np.concatenate([target, nontarget])))  # every score is a candidate threshold
    frr = np.searchsorted(target, thr, side="left") / len(target)  # targets below the threshold
    far = (len(nontarget) - np.searchsorted(nontarget, thr, side="left")) / len(nontarget)  # non-targets at or above it
    idx = int(np.argmin(np.abs(far - frr)))  # the threshold where both error rates meet
    return float((far[idx] + frr[idx]) / 2.0)


class ASVModel:
    """WeSpeaker ECAPA-TDNN wrapper: audio -> L2-normalized embedding, compared by cosine.

    Reconstructions and reference files both go through `extract_embedding_from_pcm` with the
    audio already on the model's device. wespeaker's `extract_embedding(path)` is not used: it
    computes the features on the CPU and does not move them to a CUDA model.

    This code is completely horrendous but it works... Most of the terrible code is workarounds
    for python dependency hell in order to use predefined and tested ASV models.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        # wespeaker's frontend package eagerly imports s3prl, whose old `hub` pulls in
        # every upstream and trips over torchaudio APIs removed in torchaudio 2.x
        # (set_audio_backend, sox_effects, ...). ECAPA-TDNN uses an Fbank frontend, not
        # s3prl, so stub the whole s3prl package in sys.modules: the import is satisfied
        # and the real (incompatible) zoo never loads. The S3prl frontend class is then
        # defined against dummies but never instantiated.
        import sys
        import types

        class _Stub(types.ModuleType):
            def __getattr__(self, name):
                # Dunders (__file__, __path__, __spec__, ...) must raise AttributeError so the
                # stub looks like a normal file-less module. Returning a `type` for __file__
                # hands inspect/lazy_loader a class where it expects a path string, which
                # crashes any stack walk over sys.modules, e.g. librosa's lazy `samplerate`
                # import during validation (`'__file__' has no attribute 'endswith'`).
                if name.startswith("__") and name.endswith("__"):
                    raise AttributeError(name)
                return type(name, (), {})  # any other attribute -> a throwaway dummy class

        if "s3prl" not in sys.modules:
            _s3 = _Stub("s3prl")
            _s3nn = _Stub("s3prl.nn")
            _s3.nn = _s3nn
            sys.modules["s3prl"] = _s3
            sys.modules["s3prl.nn"] = _s3nn

        import wespeaker  # lazy: only needed when validation runs

        # wespeaker's loader calls set_seed(), which reseeds the global torch RNG to 42
        # and flips on cudnn.benchmark. Both are global side effects that would perturb
        # training (RNG stream) and the variable-length validation inference (benchmark
        # re-autotunes every new length -> ~20 s/utt). Snapshot and restore them.
        rng_cpu = torch.get_rng_state()
        rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        cudnn_bench = torch.backends.cudnn.benchmark

        # A local model directory loads via load_model_local and a language code
        # (e.g. "english") via load_model. Prefer the local loader for a dir.
        if os.path.isdir(model_path) and hasattr(wespeaker, "load_model_local"):
            self.model = wespeaker.load_model_local(model_path)
        else:
            self.model = wespeaker.load_model(model_path)
        self.model.set_device(device)

        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state_all(rng_cuda)
        torch.backends.cudnn.benchmark = cudnn_bench

        self._cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _to_vec(e) -> np.ndarray:
        """A wespeaker embedding as a flat, L2-normalized float32 vector."""
        if isinstance(e, torch.Tensor):
            e = e.detach().cpu().numpy()
        e = np.asarray(e, dtype=np.float32).reshape(-1)
        return e / (np.linalg.norm(e) + 1e-9)

    def _embed(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """Embedding of a 1-D float waveform in [-1, 1].

        Adapted from wespeaker's `cli.speaker.Speaker.extract_embedding` (Apache-2.0), which
        loads audio in the int16 range (`wavform_norm=False`). The `* 32768.0` matches that
        scale, and the audio stays on the model's device (see the class docstring).
        """
        pcm = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32) * 32768.0)
        pcm = pcm.unsqueeze(0).to(self.model.device)  # (1, T)
        return self._to_vec(self.model.extract_embedding_from_pcm(pcm, sr))

    def embed_file(self, path: str) -> np.ndarray:
        """Embedding of a reference audio file, loaded at 16 kHz and cached by path."""
        v = self._cache.get(path)
        if v is None:
            wav, _ = librosa.load(path, sr=16000, mono=True)
            v = self._embed(wav, 16000)
            self._cache[path] = v
        return v

    def embed_wav(self, wav: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Embedding of an in-memory waveform, e.g. a reconstruction."""
        return self._embed(wav, sr)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity of two embeddings from this class."""
        return float(np.dot(a, b))  # both are already L2-normalized


def main():
    """Self-test of compute_eer, no model needed."""
    assert compute_eer([1.0, 0.9, 0.8], [0.1, 0.2, 0.0]) == 0.0  # fully separable
    e = compute_eer([0.5, 0.5], [0.5, 0.5])  # fully overlapping, about chance
    assert 0.4 <= e <= 0.6, e
    print("[asv.py] self-test OK")


if __name__ == "__main__":
    main()
