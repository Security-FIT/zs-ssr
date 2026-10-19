# ZS-SSR: Zero-Shot Source Speech Reconstruction

This repository contains code for reproducing our publicated results for the paper
*ZS-SSR: Zero-Shot Source Speech Reconstruction*.

**Listening.** The examples of a run are in `outputs/freevc/synth-X/examples/`.

Given a voice-converted (VC) sample, reconstruct speech in the voice of the original
*source* speaker, conditioned on a source-speaker embedding from a source-speaker
verifier. It is the inverse of voice conversion. It was built on using the data and baseline
of the [SLT 2024 Source Speaker Tracing Challenge (SSTC)](https://sstc-challenge.github.io).

```
 source speaker (LibriSpeech) ──┐
                                ├── voice conversion ──► VC sample ──────────┐
 target voice (VoxCeleb) ───────┘                                            ▼
 source-speaker verifier ──► source embedding g ──► reconstructor ──► reconstructed speech
 (embeddings/)                                      (FreeVC, AGAIN-VC,      │
                                                     DiffVC)                ▼
                                                                    ASV / EER, WER (evaluation/)
```

Although the publication reports mainly the FreeVC-based reconstructor architecture, 
four reconstructors were implemented, each an existing VC model adapted to take the
external embedding `g` instead of a reference utterance:

| Reconstructor | Package | Model | Vocoder |
|---|---|---|---|
| FreeVC | `freevc/` | Coqui's FreeVC (VITS + HiFi-GAN), WavLM content | end-to-end waveform |
| AGAIN-VC | `againvc/` | AdaIN mel auto-encoder, `g` -> AdaIN parameters | MelGAN (`torch.hub`) |
| DiffVC | `diffvc/` | frozen "average voice" encoder + diffusion decoder conditioned on `g` | HiFi-GAN |

All of them train on parallel data: the VC sample is the input, the LibriSpeech
original it was converted from is the target.

A fourth reconstructor based on [wav2vec-VC](https://github.com/prairie-schooner/wav2vec-vc) was also implemented.
We only publish its job scripts and results because the upstream code has no license.

## Results

`outputs/<reconstructor>/synth-<n>/` holds the final test results of each run:

- `val_metrics.txt`: one line per validation, the last `test-final:` line is the test result
  (EER, WER, number of test utterances `n`).
- `test-final_trials_epoch<E>.tsv`: every scored trial of the test evaluation, with the
  positive and negative ASV scores and the transcripts.
- `examples/`: listening examples.

The runs are FreeVC on Test-1..8 plus the aligned (`--align`) runs on Test-1 and Test-4, and
AGAIN-VC, DiffVC and wav2vec-VC on Test-1..8. FreeVC is the main reconstructor of the paper.
The aligned runs are a case study of the two time-shifted synthesizers (see `duration_check.py`),
made once FreeVC was chosen as the main architecture, so only FreeVC's are published.

Most runs are tested on all 13530 utterances of their test synthesizer. Exceptions:

- wav2vec-VC on Test-1, Test-2 and Test-6: 1000 utterances. The evaluation on the full set
  did not finish in the job time limit, and it was not repeated to save compute since the reconstructions
  were measurably and subjectively worse.
- DiffVC on Test-5: 2000 utterances after the last epoch (50). The full set was evaluated at
  epoch 34 (also in `val_metrics.txt`). The reconstructions sound good, but DiffVC trains much
  slower than FreeVC, so it was not repeated after FreeVC was chosen as the main architecture.

## Listening examples

Out of 200 random test utterances of each run, the three best, average and worst by the
reconstruction's ASV cosine to the source speaker, e.g. FreeVC on Test-3:

```
outputs/freevc/synth-3/examples/
├── best1_cos0.689_1vcinput.wav     VC input
├── best1_cos0.689_2recon.wav       reconstruction
├── best1_cos0.689_3source.wav      real source speaker (LibriSpeech)
├── ...                             best1..3, average1..3, worst1..3
└── manifest.json                   every triplet with its score and source files
```

## Repository layout

| Path | Contents |
|---|---|
| `train_freevc.py`, `train_againvc.py`, `train_diffvc.py` | one training entry point per reconstructor |
| `train_common.py` | flags and in-loop validation shared by the trainers |
| `freevc/` | FreeVC reconstructor: `model.py` (FreeVC with the external `g`), WavLM content backend and cache, `audio.py` / `losses.py` from FreeVC |
| `againvc/` | AGAIN-VC reconstructor, vendored from KimythAnly/AGAIN-VC |
| `diffvc/` | DiffVC reconstructor, vendored from huawei-noah/Speech-Backbones |
| `dataset/` | data code shared by all reconstructors: file name parsing and LibriSpeech lookup, (VC sample, target, `g`) pairing, tar-packed corpus, feature cache, `--align` offset estimate |
| `evaluation/` | ASV (WeSpeaker ECAPA-TDNN) and EER, Whisper WER, in-loop validation, listening examples, UTMOS22 |
| `analysis/` | paper analysis and figures, run as `python -m analysis.<script>`, outputs in `analysis/results/` and `analysis/figures/` |
| `embeddings/` | the exporter of the source-speaker embeddings and their [format](embeddings/EMBEDDINGS.md) |
| `outputs/` | published results and listening examples |
| `scripts/sge/` | our cluster job scripts ([notes](scripts/sge/README.md)) |
| `duration_check.py` | checks per synthesizer whether VC samples are time-aligned with their LibriSpeech originals |
| `pack_sltsstc.py` | packs the corpus into tars (see the appendix) |

## Setup

The steps below set up FreeVC, the main reconstructor. The other reconstructors require
extra downloads, listed in the [appendix](#appendix-other-reconstructors).

### Data

All paths default to `../datasets/` next to the repository:

```
../datasets/
├── slt-sstc/            SLT-SSTC (https://sstc-challenge.github.io), the VC inputs
│   ├── train/Train-1 .. Train-8/
│   ├── dev/Dev-1 .. Dev-12/
│   └── test/Test-1 .. Test-16/
├── LibriSpeech/         LibriSpeech (https://www.openslr.org/12), the reconstruction targets
│   ├── train-clean-100/ train-clean-360/   training source speakers
│   └── dev-clean/ test-clean/              dev and test source speakers
└── voxceleb/vox1/wav/   VoxCeleb1 (https://www.robots.ox.ac.uk/~vgg/data/voxceleb/), only for analysis/vc_quality.py
```

Index LibriSpeech once. This saves the LibriSpeech subset of every utterance to
`dataset/libri_resolve_cache.json`, so training finds each target without searching the
corpus (it also checks the layout above):

```bash
python dataset/resolver.py ../datasets/LibriSpeech --build-cache
```

### Environment

```bash
conda env create -f environment.yml      # Python 3.11, torch 2.5.1 (CUDA 12.1), coqui-tts 0.27.5
conda activate zero-shot-ssr
```

Compute nodes without internet need two downloads cached first: Whisper for the WER
(`--wer`) and the WavLM-Large content model of FreeVC.

```bash
python -c "import whisper; whisper.load_model('base.en')"
python -c "from TTS.vc.layers.freevc.wavlm import get_wavlm; get_wavlm()"
```

### Source-speaker embeddings

Every run reads the embeddings `g` and its file lists from three npz files, computed with the
MFA-Conformer of the [SSTC 2024 baseline](https://github.com/SSTC-Challenge/SSTC2024_baseline_system).
`embeddings/export_embeddings.py` runs inside a checkout of the baseline, in the baseline's own
environment (see its README):

```bash
git clone https://github.com/SSTC-Challenge/SSTC2024_baseline_system
cp embeddings/export_embeddings.py SSTC2024_baseline_system/
cd SSTC2024_baseline_system
gdown 1K5S7aUVUTqgPVgqQx7fpQWO1ZbYuJ6MX -O best.pkl     # the trained baseline model
python export_embeddings.py --checkpoint best.pkl --slt-root ../../datasets/slt-sstc \
    --save-path outputs/organizer-pretrained
mkdir -p ../embeddings/non-finetuned/organizer-pretrained
cp -r outputs/organizer-pretrained/exported_embeddings ../embeddings/non-finetuned/organizer-pretrained/
```

With recent versions of NeMo and torchaudio the baseline may need small fixes (for example,
passing the dtype to `pos_enc.extend_pe` in `modules/front_conformer_mfa.py`).

### ASV model

The reconstructions are scored by the WeSpeaker ECAPA-TDNN (VoxCeleb, large margin):

```bash
hf download Wespeaker/wespeaker-voxceleb-ecapa-tdnn1024-LM avg_model.pt config.yaml \
    --local-dir evaluation/models/wespeaker/ecapa-tdnn1024_lm
```

## Running

Train FreeVC on synthesizer 3, then evaluate on Test-3 (one GPU, ~20 GB):

```bash
EMB=embeddings/non-finetuned/organizer-pretrained/exported_embeddings
python train_freevc.py \
    --exp-npz $EMB/train_embeddings.npz --dev-npz $EMB/dev_embeddings.npz --test-npz $EMB/test_embeddings.npz \
    --train-synth Train-3 --dev-synth Dev-3 --test-synth Test-3 \
    --out-dir outputs/freevc/synth-3 --batch-size 16 --epochs 12 \
    --wavlm-cache wavlm_cache --wer --val-samples 300
```

The result lands in `outputs/freevc/synth-3/` (see [Results](#results)). The aligned runs of the
time-shifted synthesizers 1 and 4 replace `--wavlm-cache wavlm_cache` with `--align`. Other flags:

- `--wavlm-cache <dir>`: WavLM features from a per-subset cache instead of running WavLM every
  step, which makes training faster. Missing subsets are computed on the first run.
- `--resume <ckpt>`: continue training, with `--epochs 0` only rerun the final evaluation.
- `--overfit-one-batch`, `--limit-samples N`: quick checks.
- `--no-average-emb`: train on per-utterance embeddings instead of the per-speaker mean
  (dev and test always use the per-utterance embedding).

`python train_freevc.py --help` lists everything. The cluster job scripts we used are in
`scripts/sge/freevc/`.

## Data format

### SLT-SSTC file names

```
id10270 - 5r0dWxy17C8 - 00001 - 1462 - 170142 - 0034 .wav
└──┬───┘   └────┬────┘   └─┬─┘   └─────────┬─────────┘
 target      VoxCeleb    segment   LibriSpeech utterance id
 voice       video id              (reader-chapter-utterance)
```

The target voice is the VoxCeleb speaker the sample was converted into. The source speaker,
the identity we reconstruct, is the LibriSpeech reader (`1462`). `Train-N`, `Dev-N` and
`Test-N` are the same VC system for N = 1..8. Dev-9..12 and Test-9..16 have no training
counterpart.

### Embedding files

The format of the embedding files is described in [EMBEDDINGS.md](embeddings/EMBEDDINGS.md).

## License

MIT, see `LICENSE`. Vendored code keeps its own license:

- `freevc/audio.py`, `freevc/losses.py` and the training loop:
  [FreeVC](https://github.com/OlaWod/FreeVC), MIT (`freevc/LICENSE`).
- `againvc/`: [AGAIN-VC](https://github.com/KimythAnly/AGAIN-VC), MIT (`againvc/LICENSE`).
- `diffvc/`: [DiffVC](https://github.com/huawei-noah/Speech-Backbones/tree/main/DiffVC), MIT
  (`diffvc/LICENSE`). Its HiFi-GAN is from [jik876/hifi-gan](https://github.com/jik876/hifi-gan),
  MIT (`diffvc/THIRD_PARTY_NOTICE`).

The FreeVC model itself comes from [Coqui TTS](https://github.com/idiap/coqui-ai-TTS) (MPL-2.0),
installed as a dependency and used without modification, together with the WavLM-Large weights it
downloads. AGAIN-VC's MelGAN vocoder comes from
[descriptinc/melgan-neurips](https://github.com/descriptinc/melgan-neurips) via `torch.hub`.

## Appendix: other reconstructors

All of them run in the same environment and take the same flags as FreeVC
(`python train_<model>.py --help`). Our job scripts are in `scripts/sge/<model>/`.

- **AGAIN-VC** (`train_againvc.py`): the MelGAN vocoder comes from `torch.hub`. Offline nodes
  need it cached: `python -c "import torch; torch.hub.load('descriptinc/melgan-neurips', 'load_melgan', trust_repo=True)"`.
- **DiffVC** (`train_diffvc.py`): needs the pretrained encoder and HiFi-GAN vocoder of DiffVC.
  Its content encoder outputs are cached in `libritts_cache/` on the first run.

  ```bash
  gdown 1JdoC5hh7k6Nz_oTcumH0nXNEib-GDbSq -O diffvc/checkpts/enc.pt
  gdown 10khlrM645pTbQ4rc2aNEYPba8RFDBkW- -O diffvc/checkpts/vocoder/generator
  ```

- **wav2vec-VC**: the code is not published (see above), only its job scripts and results.

## Appendix: analysis

`analysis/vc_quality.py` needs VoxCeleb1 (see [Data](#data)). The naturalness scripts use
UTMOS22 from `torch.hub`, cached with
`python -c "import torch; torch.hub.load('tarepan/SpeechMOS', 'utmos22_strong', trust_repo=True)"`.

## Appendix: tar-packed corpus

SLT-SSTC is ~3M small wav files, more than some clusters allow per user. `pack_sltsstc.py`
packs each subset into one uncompressed tar plus an index (we realized too late this could be done with a simple zip):

```bash
python pack_sltsstc.py --loose-root ../datasets/slt-sstc --out-root ../datasets/slt-sstc-tar
```

Pass `--slt-root ../datasets/slt-sstc-tar` to read from the tars. The code looks for the
indices directly in `--slt-root`, so packing with `--out-root ../datasets/slt-sstc` makes the
default `--slt-root` read the tars too.

## Appendix: environment step by step

The same environment as `environment.yml`, installed with `uv`:

```bash
conda create -n zero-shot-ssr python=3.11      # Coqui needs Python >=3.10, <3.15
conda activate zero-shot-ssr
pip install uv
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
uv pip install coqui-tts 'transformers>=4.57,<5'   # coqui-tts 0.27 imports a helper removed in transformers 5
uv pip install librosa scipy numpy python_speech_features tqdm matplotlib
uv pip install git+https://github.com/wenet-e2e/wespeaker.git soundfile onnxruntime
uv pip install einops openai-whisper gdown
```
