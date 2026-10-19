#!/bin/bash
#$ -N ssr-freevc-5
#$ -q long.q
#$ -l gpu=1,gpu_ram=20G
#$ -l hostname='supergpu10|supergpu11|supergpu12|supergpu13|supergpu15|supergpu17|supergpu19|supergpu20'
#$ -o /mnt/strade/ilicka/zero-shot-ssr/outputs/freevc/synth-5/ssr-freevc-5.$JOB_ID.log
#$ -e /mnt/strade/ilicka/zero-shot-ssr/outputs/freevc/synth-5/ssr-freevc-5.$JOB_ID.log

# Before qsub (the -o/-e dir must already exist, SGE won't create it): mkdir -p /mnt/strade/ilicka/zero-shot-ssr/outputs/freevc/synth-5
source "/mnt/strade/ilicka/zero-shot-ssr/scripts/sge/_sge_header.sh"

SAVE_DIR="/mnt/strade/ilicka/zero-shot-ssr/outputs/freevc/synth-5"
mkdir -p "$SAVE_DIR"

EMB="embeddings/non-finetuned/organizer-pretrained/exported_embeddings"
# FreeVC reconstructor (from scratch, 12 epochs). WavLM content cache lives in the repo and
# is AUTO-GENERATED at startup, before epoch 0 (one WavLM pass, then WavLM is freed) for Train-5 +
# Dev-5 + Test-5.
# Stage the WavLM cache (Train-5/Dev-5/Test-5) from berta NFS to BeeGFS scratch and read it from
# there, since random reads into the npz over /mnt/strade are the per-step bottleneck.
STORAGE_CACHE="$SRCDIR/wavlm_cache"
CACHE="/mnt/scratch/tmp/ilicka/zero-shot-ssr/wavlm_cache"
mkdir -p "$STORAGE_CACHE" "$CACHE"
echo "Staging WavLM cache (Train-5/Dev-5/Test-5) to scratch at $(date)"
for s in Train-5 Dev-5 Test-5; do
    for f in "$STORAGE_CACHE/$s".npz "$STORAGE_CACHE/$s".part*.npz; do
        [ -e "$f" ] && rsync -a --info=progress2 "$f" "$CACHE/"
    done
done
python3 train_freevc.py \
        --exp-npz "$EMB/train_embeddings.npz" --dev-npz "$EMB/dev_embeddings.npz" --test-npz "$EMB/test_embeddings.npz" \
        --train-synth Train-5 --dev-synth Dev-5 --test-synth Test-5 \
        --slt-root ../shared/datasets/slt-sstc --librispeech-root ../shared/datasets/LibriSpeech \
        --out-dir "$SAVE_DIR" \
        --num-workers 4 --batch-size 16 --epochs 12 \
        --wavlm-cache "$CACHE" \
        --wer --asr-model base.en --val-samples 300 --test-samples 0
        # --resume "$SAVE_DIR/freevc_ssr_epoch<N>.pth"   # optional restart after a kill

echo "Syncing WavLM cache back to storage at $(date)"
rsync -a "$CACHE"/*.npz "$STORAGE_CACHE"/ 2>/dev/null || true

echo "Job finished at $(date)"
