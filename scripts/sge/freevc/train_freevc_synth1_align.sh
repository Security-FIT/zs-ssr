#!/bin/bash
#$ -N ssr-freevc-1-align
#$ -q long.q
#$ -l gpu=1,gpu_ram=20G
#$ -l hostname='supergpu10|supergpu11|supergpu12|supergpu13|supergpu15|supergpu17|supergpu19|supergpu20'
#$ -o /mnt/strade/ilicka/zero-shot-ssr/outputs/freevc/synth-1-align/ssr-freevc-1-align.$JOB_ID.log
#$ -e /mnt/strade/ilicka/zero-shot-ssr/outputs/freevc/synth-1-align/ssr-freevc-1-align.$JOB_ID.log

# Before qsub (the -o/-e dir must already exist, SGE won't create it): mkdir -p /mnt/strade/ilicka/zero-shot-ssr/outputs/freevc/synth-1-align
source "/mnt/strade/ilicka/zero-shot-ssr/scripts/sge/_sge_header.sh"

SAVE_DIR="/mnt/strade/ilicka/zero-shot-ssr/outputs/freevc/synth-1-align"
mkdir -p "$SAVE_DIR"

EMB="embeddings/non-finetuned/organizer-pretrained/exported_embeddings"
# FreeVC WITH --align: per-utterance correction of Train-1's constant VC/source offset.
# --align needs the raw VC waveform, so WavLM runs LIVE (no --wavlm-cache: the two are
# mutually exclusive, since the cache holds features from the UNALIGNED audio). --wer tracks WER.
# No feature cache here: the VC wavs are read loose every step. Pack the subsets this job uses
# into one INDEXED tar per subset (pack_sltsstc.py, idempotent: reuses a subset's .index.npz, else builds
# it from the loose dir), then copy them to BeeGFS scratch and read random-access from the tar:
# no unzip, no millions of loose files. Tars go in slt-sstc/tars/ (a subdir, NOT the slt-root the
# freevc/diffvc jobs read) so those stay in loose mode. LibriSpeech targets stay loose.
SLT_LOOSE="../shared/datasets/slt-sstc"
SLT_TAR="$SLT_LOOSE/tars"
SLT_ROOT="/mnt/scratch/tmp/ilicka/zero-shot-ssr/slt-sstc-tar"
mkdir -p "$SLT_TAR" "$SLT_ROOT"
echo "Packing VC tars (Train-1 Dev-1 Test-1) at $(date)"
python3 pack_sltsstc.py --loose-root "$SLT_LOOSE" --out-root "$SLT_TAR" --synth Train-1 Dev-1 Test-1
for s in Train-1 Dev-1 Test-1; do
    [ -e "$SLT_TAR/$s.index.npz" ] || { echo "ERROR: missing tar index $SLT_TAR/$s.index.npz (pack failed?)" >&2; exit 1; }
    rsync -a --info=progress2 "$SLT_TAR/$s.tar" "$SLT_TAR/$s.index.npz" "$SLT_ROOT/"
done

python3 train_freevc.py \
        --exp-npz "$EMB/train_embeddings.npz" --dev-npz "$EMB/dev_embeddings.npz" --test-npz "$EMB/test_embeddings.npz" \
        --train-synth Train-1 --dev-synth Dev-1 --test-synth Test-1 \
        --slt-root "$SLT_ROOT" --librispeech-root ../shared/datasets/LibriSpeech \
        --out-dir "$SAVE_DIR" \
        --num-workers 4 --batch-size 16 --epochs 12 \
        --align --wer --asr-model base.en --val-samples 300 --test-samples 0
        # --align-max-lag 64   # +/- frames searched for the offset

echo "Job finished at $(date)"
