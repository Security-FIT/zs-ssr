"""Speech recognition and word error rate (WER) of the reconstructions.

The EER tells whether a reconstruction recovers the source speaker, but not whether the words
survived. Misaligned or blurred content (see `duration_check.py`) hurts intelligibility while the
speaker identity, carried by the global embedding `g`, stays mostly intact. WER catches that.

* `ASRModel` transcribes a 16 kHz waveform with Whisper (`base.en` by default).
* `librispeech_transcript` gives the reference text of a LibriSpeech original.
* `word_error_rate` counts the word edits between reference and hypothesis. A corpus WER is
  sum(edits) / sum(reference words), not a mean of per-utterance WERs.

Both sides are normalized with Whisper's `EnglishTextNormalizer`, which also matches numerals to
spelled-out numbers (Whisper writes "20", LibriSpeech "TWENTY"). Whisper is imported lazily, so
`word_error_rate` and `librispeech_transcript` work without Whisper or torch installed. The
weights are downloaded to `~/.cache/whisper` on first use (offline nodes need them cached, see
README).
"""

from __future__ import annotations

import functools
import os

import numpy as np


def word_error_rate(ref: str, hyp: str, normalizer) -> tuple[int, int]:
    """(edits, reference words) between two texts, after `normalizer` (str -> str) on both.

    Edits are the word-level Levenshtein distance: the fewest substitutions, deletions and
    insertions, each costing 1, that turn the reference into the hypothesis.
    """
    ref_words, hyp_words = normalizer(ref).split(), normalizer(hyp).split()
    # prev[j] = edits turning the reference words seen so far into the first j hypothesis words,
    # only the previous row of the full table is kept
    prev = list(range(len(hyp_words) + 1))  # empty reference: j insertions
    for i, ref_word in enumerate(ref_words, 1):
        cur = [i]  # empty hypothesis: i deletions
        for j, hyp_word in enumerate(hyp_words, 1):
            cur.append(min(
                prev[j] + 1,                           # delete ref_word
                cur[j - 1] + 1,                        # insert hyp_word
                prev[j - 1] + (ref_word != hyp_word),  # substitute, free if the words match
            ))
        prev = cur
    return prev[-1], len(ref_words)


def english_normalizer():
    """Whisper's `EnglishTextNormalizer` (lowercase, no punctuation, numbers and contractions
    spelled out the same way). Every stored WER in the repo was computed with it."""
    from whisper.normalizers import EnglishTextNormalizer

    return EnglishTextNormalizer()


@functools.lru_cache(maxsize=None)
def _read_transcripts(trans_path: str) -> dict[str, str]:
    """Utterance id -> text of one LibriSpeech `.trans.txt` (lines `<utt id> THE WORDS ...`)."""
    table = {}
    with open(trans_path) as f:
        for line in f:
            uid, _, text = line.partition(" ")
            table[uid] = text.strip()
    return table


def librispeech_transcript(flac_path: str) -> str:
    """Reference text of a LibriSpeech `.flac`, from the `<reader>-<chapter>.trans.txt` next to it."""
    key = os.path.splitext(os.path.basename(flac_path))[0]  # reader-chapter-utt
    reader, chapter, _ = key.split("-")
    trans = os.path.join(os.path.dirname(flac_path), f"{reader}-{chapter}.trans.txt")
    return _read_transcripts(trans)[key]


class ASRModel:
    """Whisper transcriber. `normalize` is the text normalizer to pass to `word_error_rate`."""

    def __init__(self, model: str = "base.en", device: str = "cuda"):
        import whisper  # only loaded when a run computes WER

        self.model = whisper.load_model(model, device=device)
        self.device = device
        self.fp16 = str(device).startswith("cuda")  # Whisper warns about fp16 on CPU
        self.normalize = english_normalizer()

    def transcribe(self, wav: np.ndarray, sr: int) -> str:
        """Transcript of a mono float waveform at sample rate `sr`."""
        audio = np.ascontiguousarray(wav, dtype=np.float32)
        if sr != 16000:  # Whisper expects 16 kHz
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        out = self.model.transcribe(audio, language="en", task="transcribe", fp16=self.fp16)
        return out["text"].strip()


def main():
    """Self-test of the WER counting and normalization, no model weights needed."""
    same = str  # no normalization, compare the raw words
    assert word_error_rate("the cat sat", "the cat sat", same) == (0, 3)
    assert word_error_rate("the cat sat", "the cat sad", same) == (1, 3)      # 1 substitution
    assert word_error_rate("a b c d", "a c d", same) == (1, 4)                # 1 deletion
    assert word_error_rate("a b", "a x b y", same) == (2, 2)                  # 2 insertions
    assert word_error_rate("", "a b", same) == (2, 0)                         # empty reference
    assert word_error_rate("a b c", "", same) == (3, 3)                       # empty hypothesis
    norm = english_normalizer()
    assert word_error_rate("SIX O'CLOCK", "6 o'clock.", norm) == (0, 2)       # numerals, punctuation
    print("[asr.py] self-test OK (ASRModel needs model weights, not exercised here)")


if __name__ == "__main__":
    main()
