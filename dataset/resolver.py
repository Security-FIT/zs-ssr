"""Map SLT-SSTC VC-sample filenames to their original LibriSpeech utterances.

Shared by every reconstructor and the validation harness.

Filename grammar (SLT-SSTC):
    id10270-5r0dWxy17C8-00001-1462-170142-0034.wav
    = {VoxCeleb target voice}-{video id}-{segment}-{LibriSpeech reader}-{chapter}-{utterance}
The trailing `reader-chapter-utterance` triple identifies the LibriSpeech
original. The *reader* (first field of that triple) is the SOURCE speaker.
"""

from __future__ import annotations

import argparse
import json
import os

# Disk-persisted resolve cache.
# Maps `utt_id -> subset` (the LibriSpeech top-level dir the original is in),
# NOT a full path, so it stays valid whatever `root` is passed.
# Populated on the first run or by running this script by itself with the --build-cache flag.
DEFAULT_RESOLVE_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libri_resolve_cache.json")

# LibriSpeech subsets the SLT-SSTC source utterances come from.
SUBSETS = ("train-clean-100", "train-clean-360", "dev-clean", "test-clean")


def source_utt_id(vc_path: str) -> str:
    """LibriSpeech utterance id from a VC-sample filename = its last 3 dash fields.

    `id10270-GWXujl-xAVM-00041-3752-4943-0020.wav` -> `3752-4943-0020`
    Some VC systems append a suffix to the utterance field
    (e.g. Train-5's `...-3752-4943-0020_gen.wav`)
    Strips it so the id maps back to the LibriSpeech original `0020.flac`.
    """
    stem = os.path.splitext(os.path.basename(vc_path))[0]
    reader, chapter, utt = stem.split("-")[-3:]
    return f"{reader}-{chapter}-{utt.split('_')[0]}"


class LibriSpeechResolver:
    """Map a `reader-chapter-utt` ID to its .flac under a LibriSpeech root.

    Builds the standard path `<root>/<subset>/<reader>/<chapter>/<id>.flac` and
    probes each top-level subset dir (train-clean-100/360/...) with the reader and chapter.
    No full tree walk.
    """

    def __init__(self, root: str, ext: str = ".flac", cache_path: str | None = DEFAULT_RESOLVE_CACHE):
        self.root = root
        self.ext = ext
        self.subsets = SUBSETS
        self.cache_path = cache_path
        self._cache: dict[str, str] = {}
        self._dirty = False
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                self._cache = json.load(f)

    def path(self, utt_id: str) -> str | None:
        """Mainly checks in which LibriSpeech subdirectory the source utterance is, given an SLT-SSTC utterance.

        Returns the full path of the source utterance.
        SLT-SSTC utterance -> LibriSpeech source utterance"""
        reader, chapter, _ = utt_id.split("-")
        sub = self._cache.get(utt_id)
        if sub is not None:  # trust the cache
            return os.path.join(self.root, sub, reader, chapter, utt_id + self.ext)
        for sub in self.subsets: # check each subdirectory for the source utterance
            p = os.path.join(self.root, sub, reader, chapter, utt_id + self.ext)
            if os.path.exists(p):
                self._cache[utt_id] = sub
                self._dirty = True
                return p
        return None  # nothing found, this should be caller-side error handling.

    def save_cache(self) -> None:
        """Persist newly-resolved entries. Atomic replace.

        For sanity reasons, in practice, only one job should be ran to populate the cache.
        With others following after.

        Cache is NEVER RE-CHECKED afterwards."""
        if not (self.cache_path and self._dirty):
            return
        tmp = f"{self.cache_path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(self._cache, f)
        os.replace(tmp, self.cache_path)
        self._dirty = False


def main():
    """Self-test: the filename-grammar parse, then the LibriSpeech layout.
    With --build-cache, also (re)builds the resolve cache from the whole dataset.

        python dataset/resolver.py [LIBRISPEECH_ROOT] [--build-cache]   (default ../datasets/LibriSpeech)
    """
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default="../datasets/LibriSpeech", help="LibriSpeech root")
    p.add_argument("--build-cache", action="store_true", help=f"(re)build {DEFAULT_RESOLVE_CACHE}")
    args = p.parse_args()

    # test: the filename-grammar parse
    cases = {
        "id10270-5r0dWxy17C8-00001-1462-170142-0034.wav": "1462-170142-0034",
        "id10270-GWXujl-xAVM-00041-3752-4943-0020.wav": "3752-4943-0020",
        "train/Train-5/id10001-abc-00002-3752-4943-0020_gen.wav": "3752-4943-0020",
    }
    for path, want in cases.items():
        got = source_utt_id(path)
        assert got == want, f"{path}: {got!r} != {want!r}" # !r calls the repr() function on the variable
    print("[resolver.py] source_utt_id OK:", cases)

    # test: the LibriSpeech subsets the SLT-SSTC source speakers come from are in place,
    # and one real utterance per subset resolves (cache bypassed)
    print("[resolver.py] checking the LibriSpeech structure.")
    root = args.root
    missing = [s for s in SUBSETS if not os.path.isdir(os.path.join(root, s))]
    assert not missing, f"{root}: missing LibriSpeech subset(s) {missing}"
    resolver = LibriSpeechResolver(root, cache_path=None)
    for subset in SUBSETS:
        reader = min(os.listdir(os.path.join(root, subset)))
        chapter = min(os.listdir(os.path.join(root, subset, reader)))
        flac = min(f for f in os.listdir(os.path.join(root, subset, reader, chapter)) if f.endswith(".flac"))
        want = os.path.join(root, subset, reader, chapter, flac)
        got = resolver.path(os.path.splitext(flac)[0])
        assert got == want, f"{flac}: {got!r} != {want!r}"
    print(f"[resolver.py] LibriSpeech layout OK under {root}: {', '.join(SUBSETS)}")

    if not args.build_cache:
        return
    print("[resolver.py] Building the SLT-SSTC -> LibriSpeech source utterance mapping cache.")
    # every utterance of every subset, from directory listings (one per chapter, no per-file stat)
    resolver = LibriSpeechResolver(root)
    resolver._cache = {}
    for subset in SUBSETS:
        for _, _, files in os.walk(os.path.join(root, subset)):
            for f in files:
                if f.endswith(resolver.ext):
                    resolver._cache[f[: -len(resolver.ext)]] = subset
    resolver._dirty = True
    resolver.save_cache()
    print(f"[resolver.py] cached {len(resolver._cache)} utterances -> {resolver.cache_path}")

if __name__ == "__main__":
    main()
