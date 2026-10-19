#!/usr/bin/env python3
"""Export source-speaker embeddings of the whole SLT-SSTC corpus and score them.

Runs the trained MFA-Conformer of the SSTC 2024 baseline
(https://github.com/SSTC-Challenge/SSTC2024_baseline_system) over every utterance of every
VC subset, one utterance per forward pass like the baseline's `scoring.py`, and writes the
protocol files the reconstructors read (see EMBEDDINGS.md):

    <save_path>/exported_embeddings/
        train_embeddings.npz   file_paths, embeddings, synthesizer, speaker_label, source_speaker
        dev_embeddings.npz     file_paths, embeddings, synthesizer
        test_embeddings.npz    file_paths, embeddings, synthesizer
        manifest.json
    <save_path>/metrics.txt

metrics.txt holds the per-synthesizer and overall EER of the dev/test trial lists and the
closed-set source-speaker accuracy on train (cosine to the checkpoint's ArcFace class centers).
`--metrics-only` recomputes them from already exported npz files.

The script imports the baseline's model, data loading and EER code, so copy it into the root
of a baseline checkout and run it there, in the baseline's environment:

    python export_embeddings.py --checkpoint best.pkl --slt-root ../datasets/slt-sstc \\
        --save-path outputs/organizer-pretrained
"""
import argparse, json, os
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from scipy import spatial

import modules.model_spk as models
from dataset import WavDataset, logFbankCal
from tools.util import compute_eer

SPLIT_PREFIX = {"train": "Train-", "dev": "Dev-", "test": "Test-"}


def build_spk2int(spk2utt_path, order):
    """Source speaker -> class index of the classifier head, in spk2utt order or sorted."""
    spks = [l.split()[0] for l in open(spk2utt_path)]
    if order == "sorted":
        spks = sorted(set(spks), key=lambda s: (len(s), s))  # numeric order of the reader ids
    return {s: i for i, s in enumerate(spks)}


def list_split(slt_root, split, synth_filter, spk2int):
    """All wavs of a split as dicts {rel, abs, synth, src, label}, ordered by subset and file name.

    `src` (the LibriSpeech reader, field -3 of the file name) and `label` are only set for train.
    """
    base = Path(slt_root) / split
    dirs = sorted((d for d in base.glob(SPLIT_PREFIX[split] + "*") if d.is_dir()),
                  key=lambda p: int(p.name.split("-")[-1]))  # skips the Dev-N.zip archives
    items = []
    for d in dirs:
        if synth_filter and d.name not in synth_filter:
            continue
        for f in sorted(os.scandir(d), key=lambda e: e.name):
            if not f.name.endswith(".wav"):
                continue
            src = f.name[:-4].split("-")[-3] if split == "train" else None
            items.append({"rel": f"{split}/{d.name}/{f.name}", "abs": f.path,
                          "synth": d.name, "src": src,
                          "label": spk2int.get(src, -1) if src is not None else -1})
    return items


def load_model(ckpt_path, model_name, model_cfg, embd_dim_arg):
    """Encoder from a training checkpoint. Returns (model, embd_dim, classifier weight or None).

    The embedding size is read from the checkpoint's bottleneck, since the baseline's train.py
    default (128) differs from the released model (256).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}  # strip DataParallel prefix
    embd_dim = embd_dim_arg
    for k, v in sd.items():
        if k.endswith("bottleneck.weight"):
            embd_dim = v.shape[0]
            break
    cfg = OmegaConf.load(model_cfg)
    cfg["encoder"]["mask_prob"] = 0
    model = getattr(models, model_name)(cfg["encoder"], embd_dim, dropout=0)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[export_embeddings.py] missing={len(missing)} unexpected={len(unexpected)} keys "
              f"(first missing: {missing[:2]}, first unexpected: {unexpected[:2]})")
    cls_w = None
    if isinstance(ckpt, dict) and isinstance(ckpt.get("classifier"), dict):
        cw = ckpt["classifier"].get("weight")
        if cw is not None:
            cls_w = cw.float()
    return model, embd_dim, cls_w


def encode(model, featCal, items, fs, workers):
    """[N, D] float32 embeddings, one forward pass per utterance."""
    ds = WavDataset([[it["rel"], it["abs"]] for it in items], fs=fs)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=workers, pin_memory=True)
    embs, n = [], len(items)
    with torch.no_grad():
        for j, (feat, _utt) in enumerate(loader):
            feat = feat.cuda()
            feat_len = torch.LongTensor([feat.shape[1]]).cuda()  # length in samples, as in scoring.py
            embs.append(model(featCal(feat), feat_len).cpu().numpy()[0])
            if (j + 1) % 5000 == 0:
                print(f"[export_embeddings.py] encoded {j + 1}/{n}", flush=True)
    return np.stack(embs).astype("float32")


def write_npz(out_dir, split, items, embeddings):
    arrs = dict(
        file_paths=np.array([it["rel"] for it in items], dtype=object),
        embeddings=embeddings,
        synthesizer=np.array([it["synth"] for it in items], dtype=object),
    )
    if split == "train":
        arrs["speaker_label"] = np.array([it["label"] for it in items], dtype=np.int64)
        arrs["source_speaker"] = np.array([it["src"] for it in items], dtype=object)
    path = out_dir / f"{split}_embeddings.npz"
    np.savez(path, **arrs)
    print(f"[export_embeddings.py] {split}: wrote {len(items)} embeddings -> {path}")


def eer_from_trials(trials_path, npz_path):
    """{synthesizer: (EER %, minDCF, n trials)} plus an OVERALL entry, for one split.

    Trial lines are `<key> <utt1> <utt2>` (key 1 = same source speaker) with bare utterance
    ids: no folder, no `.wav`, no `_gen` suffix. Every subset of the split (Dev-1, Dev-2, ...)
    reuses the same pairs against its own embeddings.
    """
    d = np.load(npz_path, allow_pickle=True)
    by_synth = {}  # synth -> {utterance id: embedding}
    for rel, emb in zip(d["file_paths"], d["embeddings"]):
        stem = rel.split("/")[-1][:-4]
        if stem.endswith("_gen"):
            stem = stem[:-4]
        by_synth.setdefault(rel.split("/")[1], {})[stem] = emb
    trials = [ln.split() for ln in open(trials_path)]
    results, all_t, all_n = {}, [], []
    for synth in sorted(by_synth, key=lambda s: int(s.split("-")[-1])):
        emb = by_synth[synth]
        tg, ng = [], []
        for key, u1, u2 in trials:
            a, b = emb.get(u1), emb.get(u2)
            if a is None or b is None:
                continue
            score = 1 - spatial.distance.cosine(a, b)
            (tg if key == "1" else ng).append(score)
        all_t += tg; all_n += ng
        if tg and ng:
            eer, _, mindcf, _ = compute_eer(np.array(tg), np.array(ng))
            results[synth] = (eer * 100, mindcf, len(tg) + len(ng))
    if all_t and all_n:
        eer, _, mindcf, _ = compute_eer(np.array(all_t), np.array(all_n))
        results["OVERALL"] = (eer * 100, mindcf, len(all_t) + len(all_n))
    return results


def accuracy_train(embeddings, labels, cls_w):
    """Closed-set source-speaker accuracy via cosine to the ArcFace class centers."""
    if cls_w is None:
        return None
    W = F.normalize(cls_w.cuda(), dim=1)
    correct = total = 0
    for i in range(0, len(embeddings), 8192):
        e = F.normalize(torch.from_numpy(embeddings[i:i + 8192]).cuda(), dim=1)
        pred = (e @ W.t()).argmax(1).cpu().numpy()
        lab = labels[i:i + 8192]
        m = lab >= 0  # speakers missing from spk2utt are not scored
        correct += int((pred[m] == lab[m]).sum())
        total += int(m.sum())
    return correct / total if total else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--slt-root", default="../datasets/slt-sstc")
    ap.add_argument("--save-path", required=True, help="npz go to <save_path>/exported_embeddings/")
    ap.add_argument("--dev-trials", default=None, help="default: <slt_root>/dev/dev_trials")
    ap.add_argument("--test-trials", default=None, help="default: <slt_root>/test_trials-with_target")
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    ap.add_argument("--synthesizers", nargs="+", default=None, help="restrict to e.g. Train-1 Dev-3")
    ap.add_argument("--spk2utt", default="data/librispeech/train/spk2utt")
    ap.add_argument("--spk-order", choices=["spk2utt", "sorted"], default="spk2utt")
    ap.add_argument("--model", default="ConformerMFA")
    ap.add_argument("--model-cfg", default="configs/conformer_small2.yaml")
    ap.add_argument("--embd-dim", default=256, type=int, help="fallback if not inferable from ckpt")
    ap.add_argument("--fs", default=16000, type=int)
    ap.add_argument("--fft", default=512, type=int)
    ap.add_argument("--mels", default=80, type=int)
    ap.add_argument("--win-len", default=0.025, type=float)
    ap.add_argument("--hop-len", default=0.01, type=float)
    ap.add_argument("--workers", default=4, type=int)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--metrics-only", action="store_true", help="recompute EER/acc from existing npz")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    slt_root = Path(args.slt_root).resolve()
    out_dir = Path(args.save_path).resolve() / "exported_embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    spk2int = build_spk2int(args.spk2utt, args.spk_order)
    dev_trials = args.dev_trials or slt_root / "dev" / "dev_trials"
    test_trials = args.test_trials or slt_root / "test_trials-with_target"

    cls_w = None
    if not args.metrics_only:
        featCal = logFbankCal(sample_rate=args.fs, n_fft=args.fft,
                              win_length=int(args.win_len * args.fs),
                              hop_length=int(args.hop_len * args.fs), n_mels=args.mels).cuda()
        featCal.eval()
        model, embd_dim, cls_w = load_model(args.checkpoint, args.model, args.model_cfg, args.embd_dim)
        model = model.cuda().eval()
        print(f"[export_embeddings.py] {args.model} embd_dim={embd_dim} "
              f"classifier_head={'yes' if cls_w is not None else 'no'}")

        counts = {}
        for split in args.splits:
            items = list_split(slt_root, split, args.synthesizers, spk2int)
            print(f"[export_embeddings.py] {split}: {len(items)} utterances", flush=True)
            emb = encode(model, featCal, items, args.fs, args.workers)
            write_npz(out_dir, split, items, emb)
            counts[split] = len(items)
        manifest = dict(checkpoint=str(Path(args.checkpoint).resolve()), model=args.model,
                        embd_dim=int(embd_dim), splits=args.splits, counts=counts,
                        synthesizers=args.synthesizers, spk_order=args.spk_order,
                        n_speaker_classes=len(spk2int))
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    else:
        # accuracy still needs the classifier head
        _, _, cls_w = load_model(args.checkpoint, args.model, args.model_cfg, args.embd_dim)

    lines = []
    for split, trials in (("dev", dev_trials), ("test", test_trials)):
        npz = out_dir / f"{split}_embeddings.npz"
        if not npz.exists() or not Path(trials).exists():
            continue
        results = eer_from_trials(trials, npz)
        lines.append(f"== {split} EER (trials: {trials}) ==")
        for synth, (eer, mindcf, n) in results.items():
            lines.append(f"  {synth:>9}: EER {eer:6.2f}%   minDCF {mindcf:.4f}   ({n} trials)")

    train_npz = out_dir / "train_embeddings.npz"
    if train_npz.exists() and cls_w is not None:
        d = np.load(train_npz, allow_pickle=True)
        # labels from source_speaker, so --spk-order also applies with --metrics-only
        labels = np.array([spk2int.get(s, -1) for s in d["source_speaker"]], dtype=np.int64)
        acc = accuracy_train(d["embeddings"], labels, cls_w)
        if acc is not None:
            lines.append(f"== train source-speaker accuracy: {acc*100:.2f}%  (spk_order={args.spk_order}) ==")

    report = "\n".join(lines)
    print(report)
    (out_dir.parent / "metrics.txt").write_text(report + "\n")


if __name__ == "__main__":
    main()
