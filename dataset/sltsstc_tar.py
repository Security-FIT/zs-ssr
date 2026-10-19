"""Tar-packed SLT-SSTC corpus: pack millions of VC wavs into one random-access tar per subset.

The SLT-SSTC corpus is ~3M tiny `.wav` files: an inode problem on quota-limited
filesystems (e.g. LUMI / merlin, ~2M-file caps). This packs each VC subset's wavs into a
single uncompressed tar (`Train-3.tar`) plus a small per-subset index
(`Train-3.index.npz`: `file_paths, offset, size` + the tar name). ~3M files -> a couple
dozen. Uncompressed = the member bytes are the original wav verbatim, so a read is a plain
`seek`+`read` (no tar/inflate overhead), so random access stays fast, and the tar is still
standard (untar anywhere).

- Pack on the machine that has the loose corpus (`pack_sltsstc.py`, a heavy NFS read,
  optionally one `--synth` subset per job).
- Read transparently: entry points call `activate(slt_root)` once, if `slt_root` holds
  `*.index.npz`, every `read_wav()` of a path under it is served from the tars, otherwise
  `read_wav()` is plain librosa on the loose file.

Fork-safe for DataLoader workers: the per-subset index is loaded lazily and tar file
handles are opened per process (reset on PID change).
"""

from __future__ import annotations

import glob
import io
import os
import tarfile

import numpy as np

from dataset.npz_cache import subset_of_fp

INDEX_SUFFIX = ".index.npz"


# pack
def _subset_dirs(loose_root: str, synthesizers=None):
    """Map subset name -> its loose directory under <loose_root>/{train,dev,test}/<Subset-N>."""
    want = set(synthesizers) if synthesizers else None
    out = {}
    for split in ("train", "dev", "test"):
        split_dir = os.path.join(loose_root, split)
        if not os.path.isdir(split_dir):
            continue
        for sub in sorted(os.listdir(split_dir)):
            d = os.path.join(split_dir, sub)
            if os.path.isdir(d) and (want is None or sub in want):
                out[sub] = (split, d)
    return out


def pack(loose_root: str, out_root: str, synthesizers=None):
    """Pack each subset's wavs into one uncompressed tar + a per-subset index npz.

    Member names are the dataset-relative path `<split>/<subset>/<file>` (so the tar untars
    to the original layout). Idempotent per subset: skips a subset whose index npz already
    exists. Run where the loose corpus lives. Reading every file once is the slow part, so split
    the work across jobs with `synthesizers`.
    """
    os.makedirs(out_root, exist_ok=True)
    subsets = _subset_dirs(loose_root, synthesizers)
    print(f"[pack] {len(subsets)} subset(s): {', '.join(subsets)} -> {out_root}")
    for subset, (split, d) in subsets.items():
        if os.path.exists(os.path.join(out_root, subset + INDEX_SUFFIX)):
            print(f"[pack] {subset}: index exists — skip")
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith(".wav"))
        tar_name = _write_tar(split, subset, d, files, out_root)
        _write_index(out_root, subset, tar_name)
        print(f"[pack] {subset}: {len(files)} wavs -> {tar_name}")


def _write_tar(split, subset, subset_dir, files, out_root):
    """Write all of a subset's wavs into one uncompressed `<subset>.tar`, return its name."""
    name = f"{subset}.tar"
    with tarfile.open(os.path.join(out_root, name), "w") as tar:  # uncompressed
        for fname in files:
            arc = f"{split}/{subset}/{fname}"  # dataset-relative fp = the member name
            tar.add(os.path.join(subset_dir, fname), arcname=arc)
    return name


def _write_index(out_root, subset, tar_name):
    """Scan the finished tar (tarfile sets offset_data reliably on READ) -> index npz."""
    fps, offset, size = [], [], []
    with tarfile.open(os.path.join(out_root, tar_name), "r") as tar:
        for m in tar:
            if not m.isfile():
                continue
            fps.append(m.name)
            offset.append(m.offset_data)
            size.append(m.size)
    np.savez(
        os.path.join(out_root, subset + INDEX_SUFFIX),
        tar=np.array(tar_name),
        file_paths=np.array(fps),
        offset=np.array(offset, dtype=np.int64),
        size=np.array(size, dtype=np.int64),
    )


# read
class CorpusArchive:
    """Random-access reader over the per-subset tars. `load(path, sr)` -> 1-D wav.

    `root` holds `<subset>.tar` + `<subset>.index.npz`. Content paths are `<root>/<fp>`
    (pass `--slt-root <root>`), so `load` recovers `fp` by `relpath`.
    """

    def __init__(self, root: str):
        self.root = root
        self._pid: int | None = None
        self._tar: dict[str, str] = {}        # subset -> tar filename
        self._index: dict[str, dict] = {}     # subset -> {fp: (offset, size)}
        self._fh: dict = {}                   # subset -> open binary file handle

    @classmethod
    def open(cls, slt_root: str) -> "CorpusArchive | None":
        """Archive if `slt_root` is tar-packed (has any `*.index.npz`), else None (loose)."""
        return cls(slt_root) if glob.glob(os.path.join(slt_root, "*" + INDEX_SUFFIX)) else None

    def owns(self, path: str) -> bool:
        """Whether `path` is under this archive's root (a VC content path vs a target path)."""
        root = os.path.abspath(self.root)
        p = os.path.abspath(path)
        return p == root or p.startswith(root + os.sep)

    def _reset_if_forked(self):
        pid = os.getpid()
        if pid != self._pid:  # forked worker: drop inherited handles + caches, rebuild ours
            for fh in self._fh.values():
                try:
                    fh.close()
                except Exception:
                    pass
            self._index.clear()
            self._tar.clear()
            self._fh.clear()
            self._pid = pid

    def _ensure_index(self, subset: str):
        if subset in self._index:
            return
        idx_path = os.path.join(self.root, subset + INDEX_SUFFIX)
        if not os.path.exists(idx_path):
            raise FileNotFoundError(f"no tar index for subset {subset!r} under {self.root}")
        d = np.load(idx_path, allow_pickle=True)
        self._tar[subset] = str(d["tar"])
        self._index[subset] = {
            str(fp): (int(off), int(sz))
            for fp, off, sz in zip(d["file_paths"], d["offset"], d["size"])
        }

    def _handle(self, subset: str):
        fh = self._fh.get(subset)
        if fh is None:
            fh = open(os.path.join(self.root, self._tar[subset]), "rb")
            self._fh[subset] = fh
        return fh

    def read_bytes(self, fp: str) -> bytes:
        """Raw original-file bytes for a dataset-relative path (verbatim from the tar)."""
        self._reset_if_forked()
        subset = subset_of_fp(fp)
        self._ensure_index(subset)
        rec = self._index[subset].get(fp)
        if rec is None:
            raise KeyError(f"{fp!r} not in tar archive for {subset!r} ({self.root})")
        offset, size = rec
        fh = self._handle(subset)
        fh.seek(offset)
        return fh.read(size)

    def load(self, path: str, sr: int = 16000, mono: bool = True) -> np.ndarray:
        """Load a VC wav given a `<root>/<fp>` path (mirrors `librosa.load(path, sr, mono)`)."""
        return self.load_fp(os.path.relpath(path, self.root), sr, mono)

    def load_fp(self, fp: str, sr: int = 16000, mono: bool = True) -> np.ndarray:
        import soundfile as sf

        data, sr0 = sf.read(io.BytesIO(self.read_bytes(fp)), dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(1) if mono else data
        if sr0 != sr:
            import librosa

            data = librosa.resample(data, orig_sr=sr0, target_sr=sr)
        return np.ascontiguousarray(data, dtype=np.float32)


# process-wide active archive (set once at entry-point startup)
# entry points call activate() before spawning DataLoader workers, and every wav loader then
# goes through read_wav(), which picks the tar or the loose file by path
_ACTIVE: "CorpusArchive | None" = None


def activate(slt_root: str) -> "CorpusArchive | None":
    """Register the corpus for this process. If `slt_root` is tar-packed, later read_wav()
    calls for paths under it hit the tars, otherwise read_wav() is plain librosa. Call once
    at entry-point startup, before creating DataLoader workers."""
    global _ACTIVE
    _ACTIVE = CorpusArchive.open(slt_root)
    print(f"[sltsstc_tar] corpus at {slt_root}: {'TAR-packed (random-access)' if _ACTIVE else 'loose files'}")
    return _ACTIVE


def read_wav(path: str, sr: int = 16000, mono: bool = True) -> np.ndarray:
    """Load a wav, from the active tar archive if `path` is under it, else the loose file.

    Drop-in for `librosa.load(path, sr=sr, mono=mono)[0]`. A target (LibriSpeech) path is
    never under the SLT-SSTC archive root, so it always falls through to librosa.
    """
    if _ACTIVE is not None and _ACTIVE.owns(path):
        return _ACTIVE.load(path, sr, mono)
    import librosa

    return librosa.load(path, sr=sr, mono=mono)[0]


def main():
    # smoke test: pack fake files, verify the index + raw seek/read round-trip (no audio deps).
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        loose = os.path.join(tmp, "slt")
        out = os.path.join(tmp, "tar")
        payloads = {}
        for split, sub in (("train", "Train-9"), ("dev", "Dev-9")):
            sd = os.path.join(loose, split, sub)
            os.makedirs(sd)
            for i in range(4):
                fp = f"{split}/{sub}/id{i}-vid-000-100-200-{i:04d}.wav"
                data = bytes([(i * 37 + j) % 256 for j in range(500 + i * 111)])  # unique per file
                with open(os.path.join(loose, fp), "wb") as f:
                    f.write(data)
                payloads[fp] = data

        pack(loose, out)
        assert os.path.exists(os.path.join(out, "Train-9.tar")), "expected one <subset>.tar"
        arch = CorpusArchive.open(out)
        assert arch is not None, "tar mode not detected"
        for fp, data in payloads.items():
            assert arch.read_bytes(fp) == data, fp              # by bare fp
            assert arch.read_bytes(fp) == arch.read_bytes(fp)   # idempotent
        # path form (as datasets call it): <root>/<fp>
        any_fp = next(iter(payloads))
        assert arch.read_bytes(os.path.relpath(os.path.join(out, any_fp), out)) == payloads[any_fp]
        assert CorpusArchive.open(loose) is None, "loose corpus wrongly detected as tar"
        assert subset_of_fp(any_fp) in ("Train-9", "Dev-9")
    print("[sltsstc_tar] OK: pack tar + index, random-access read round-trip, loose/tar detection")


if __name__ == "__main__":
    main()
