"""Shared, model-agnostic data code for the source-speech reconstructors.

- `resolver`: SLT-SSTC filename grammar -> source speaker + LibriSpeech original.
- `pairing` : exported npz -> (vc_wav, target_wav, g) triples + speaker tables.
- `sltsstc_tar`: tar-packed corpus (pack + random-access `CorpusArchive`) so the
  ~3M-file loose corpus fits under strict inode caps.
- `npz_cache`: per-subset `.npz` feature cache (FreeVC WavLM features, DiffVC encoder mels).
- `align`: constant VC/source time-offset estimate behind `--align`.

Per-model feature extraction (WavLM features / mels) lives in each reconstructor
package (freevc/, againvc/, diffvc/).
"""

from dataset.pairing import build_parallel_samples, build_speaker_embedding_table
from dataset.resolver import LibriSpeechResolver, source_utt_id
from dataset.sltsstc_tar import CorpusArchive

__all__ = [
    "LibriSpeechResolver",
    "source_utt_id",
    "build_parallel_samples",
    "build_speaker_embedding_table",
    "CorpusArchive",
]
