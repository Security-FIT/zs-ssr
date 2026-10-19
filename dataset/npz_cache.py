"""Per-subset `.npz` cache of per-utterance features (FreeVC WavLM features, DiffVC encoder mels).

One `<cache_dir>/<subset>.npz` per VC subset (e.g. `Train-3.npz`) instead of one file per
utterance, so a ~327k-utterance subset costs one inode. Each member holds one utterance's
feature, keyed by its dataset-relative path (`train/Train-3/id....wav`). Members are stored
uncompressed (`ZIP_STORED`), so reading one is a seek into the file.

Written streaming, one utterance in memory at a time (a train subset is ~100 GB), and read
lazily through numpy's `NpzFile`, which loads only the requested member.
"""

from __future__ import annotations

import io
import os
import zipfile

import numpy as np


def subset_of_fp(fp: str) -> str:
    """VC subset of a dataset-relative path: `train/Train-3/id....wav` -> `Train-3` (the parent dir)."""
    parts = fp.replace("\\", "/").rstrip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"cannot derive VC subset from {fp!r} (expected .../<Subset-N>/<file>)")
    return parts[-2]


def subset_file_paths(npz_paths, synths=None) -> dict[str, list[str]]:
    """Group the `file_paths` of exported embedding npz(s) by VC subset: {subset: [fp, ...]}.

    `synths` (e.g. `["Train-3"]`) keeps only those subsets, None keeps all.
    """
    want = set(synths) if synths else None
    out: dict[str, list[str]] = {}
    for npz in npz_paths:
        d = np.load(npz, allow_pickle=True)
        for fp in d["file_paths"]:
            subset = subset_of_fp(str(fp))
            if want is None or subset in want:
                out.setdefault(subset, []).append(str(fp))
    return out


def write_subset(cache_dir: str, subset: str, items) -> str | None:
    """Stream `(fp, feature)` pairs into `<cache_dir>/<subset>.npz` and return its path.

    Skips (returns None) when the file already exists. Written to a temp file and renamed
    into place, so a killed job never leaves a partial npz behind.
    """
    os.makedirs(cache_dir, exist_ok=True)
    out = os.path.join(cache_dir, subset + ".npz")
    if os.path.exists(out):
        print(f"[npz_cache.py] {out} exists, skip")
        return None
    tmp = f"{out}.tmp.{os.getpid()}"
    n = 0
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for fp, feature in items:
                buf = io.BytesIO()
                np.save(buf, np.ascontiguousarray(feature, dtype=np.float32))
                zf.writestr(fp + ".npy", buf.getvalue())  # NpzFile keys the member by `fp`
                n += 1
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    print(f"[npz_cache.py] wrote {n} features -> {out}")
    return out


class NpzCache:
    """Read side: a VC sample's path `<slt_root>/<fp>` -> its cached feature.

    File handles are opened lazily and per process: an `NpzFile` wraps a single file handle,
    so each DataLoader worker reopens its own after the fork.
    """

    def __init__(self, cache_dir: str, slt_root: str):
        self.cache_dir = cache_dir
        self.slt_root = slt_root
        self._pid: int | None = None
        self._handles: dict[str, np.lib.npyio.NpzFile] = {}  # subset -> open npz
        self._keys: dict[str, set[str]] = {}                  # subset -> member keys

    def path_for(self, subset: str) -> str:
        return os.path.join(self.cache_dir, subset + ".npz")

    def has(self, subset: str) -> bool:
        return os.path.exists(self.path_for(subset))

    def _open(self, subset: str) -> None:
        if os.getpid() != self._pid:  # forked worker: drop the inherited handles
            self._handles.clear()
            self._keys.clear()
            self._pid = os.getpid()
        if subset in self._handles:
            return
        if not self.has(subset):
            raise FileNotFoundError(f"no cache for subset {subset!r} under {self.cache_dir}")
        handle = np.load(self.path_for(subset), allow_pickle=False)
        self._handles[subset] = handle
        self._keys[subset] = set(handle.files)  # reads only the zip directory

    def get(self, path: str) -> np.ndarray:
        """Cached feature of the VC sample at `<slt_root>/<fp>`."""
        fp = os.path.relpath(path, self.slt_root)
        subset = subset_of_fp(fp)
        self._open(subset)
        if fp not in self._keys[subset]:
            raise KeyError(f"{fp!r} not in {self.path_for(subset)}")
        return self._handles[subset][fp]

    def feature_dim(self, subset: str) -> int:
        """First axis of any cached feature of `subset` (features are `(dim, T)`)."""
        self._open(subset)
        return int(self._handles[subset][next(iter(self._keys[subset]))].shape[0])


def main():
    # test: write a fake subset, skip on rewrite, read it back, group an embeddings-style npz
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        slt_root, cache_dir = os.path.join(d, "slt"), os.path.join(d, "cache")
        fps = [f"train/Train-9/id{i}-v-000-100-200-{i:04d}.wav" for i in range(4)]
        feats = {fp: np.random.randn(8, 5 + i).astype(np.float32) for i, fp in enumerate(fps)}
        assert write_subset(cache_dir, "Train-9", ((fp, feats[fp]) for fp in fps))
        assert write_subset(cache_dir, "Train-9", iter(())) is None
        cache = NpzCache(cache_dir, slt_root)
        assert cache.has("Train-9") and cache.feature_dim("Train-9") == 8
        for fp in fps:
            assert np.array_equal(cache.get(os.path.join(slt_root, fp)), feats[fp]), fp

        emb_npz = os.path.join(d, "emb.npz")
        np.savez(emb_npz, file_paths=np.array(fps + ["dev/Dev-1/z-a-b-1-2-0001.wav"]))
        assert subset_file_paths([emb_npz]) == {"Train-9": fps, "Dev-1": ["dev/Dev-1/z-a-b-1-2-0001.wav"]}
        assert subset_file_paths([emb_npz], ["Train-9"]) == {"Train-9": fps}
    print("[npz_cache.py] OK: streaming write, skip-existing, lazy read, subset grouping")


if __name__ == "__main__":
    main()
