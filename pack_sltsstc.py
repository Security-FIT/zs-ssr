"""Pack the loose SLT-SSTC corpus into one tar per subset + per-subset indices (see dataset/sltsstc_tar.py).

Run where the loose corpus lives (MetaCentrum). Reads every wav once (the slow part:
~3M tiny files on NFS), so split the work across jobs with `--synth` (one or a few
subsets per job). Idempotent per subset (skips a subset whose index npz exists).

    python pack_sltsstc.py --loose-root ../datasets/slt-sstc --out-root ../datasets/slt-sstc-tar
    python pack_sltsstc.py --loose-root ../datasets/slt-sstc --out-root ../datasets/slt-sstc-tar --synth Train-3
"""

from __future__ import annotations

import argparse

from dataset.sltsstc_tar import pack


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--loose-root", default="../datasets/slt-sstc", help="the loose corpus root (train/ dev/ test/)")
    p.add_argument("--out-root", required=True, help="output dir for the tars + index npz")
    p.add_argument("--synth", nargs="+", default=None, help="only pack these subset(s), e.g. Train-3 Dev-3 Test-3 (default: all)")
    args = p.parse_args()
    pack(args.loose_root, args.out_root, synthesizers=args.synth)
    print("[pack] done")


if __name__ == "__main__":
    main()
