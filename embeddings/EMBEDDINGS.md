# Source-speaker embeddings

The reconstructors take the source speaker's identity as an embedding `g` from a
source-speaker verifier. The embeddings are computed once for the whole SLT-SSTC corpus and
stored as one `.npz` per split, which also serves as the data protocol: every file the
reconstructors train or evaluate on is listed there.

## Producing them

[`export_embeddings.py`](export_embeddings.py) runs the MFA-Conformer source-speaker
verifier of the SSTC 2024 baseline over every wav of every VC subset (one utterance per
forward pass, the baseline's own preprocessing) and writes:

```
<save_path>/
├── exported_embeddings/
│   ├── train_embeddings.npz
│   ├── dev_embeddings.npz
│   ├── test_embeddings.npz
│   └── manifest.json    # checkpoint, model, embedding dim, utterance count per split
└── metrics.txt          # dev/test EER per subset, train source-speaker accuracy
```

The script uses the model, data loading and EER code of
[SSTC2024_baseline_system](https://github.com/SSTC-Challenge/SSTC2024_baseline_system), so it
runs inside a checkout of that repository. The commands are in the main README, section
"Source-speaker embeddings". The trained baseline model is
[here](https://drive.google.com/file/d/1K5S7aUVUTqgPVgqQx7fpQWO1ZbYuJ6MX/view?usp=drive_link).
The exported embeddings give the dev EERs reported in the baseline's README (e.g. 7.67% on
Dev-3).

## File format

All arrays in a file are row-aligned: row `i` of every array is the same utterance. There is
one row per wav file.

| key | splits | shape | meaning |
|-----|--------|-------|---------|
| `file_paths` | all | `[N]` str | wav path relative to the SLT-SSTC root, e.g. `train/Train-3/id...-0040.wav` |
| `embeddings` | all | `[N, D]` float32 | verifier embedding (`D = 256`), not L2-normalized |
| `synthesizer` | all | `[N]` str | VC subset, e.g. `Train-3` |
| `source_speaker` | train | `[N]` str | LibriSpeech reader of the source utterance, e.g. `296` |
| `speaker_label` | train | `[N]` int64 | class index of `source_speaker` in the baseline's classifier head (`-1` if unknown) |

dev/test have no per-utterance speaker label, their evaluation protocol only defines
trial-level same/different labels. The string arrays are object arrays, so load with
`allow_pickle=True`:

```python
import numpy as np

d = np.load("exported_embeddings/test_embeddings.npz", allow_pickle=True)
emb_by_path = dict(zip(d["file_paths"], d["embeddings"]))
vec = emb_by_path["test/Test-3/id10005-1geDB-I2TjE-00001-2094-142345-0058.wav"]
```

## Matching the corpus

`file_paths` mirror the SLT-SSTC directory layout, so `os.path.join(slt_root, file_path)` is the
wav. Each `Train-N` / `Dev-N` / `Test-N` folder is one VC system. Select subsets by filtering on
`synthesizer` (the trainers do this with `--train-synth` / `--dev-synth` / `--test-synth`).
