#!/bin/bash
#$ -N ssr-diffvc-5
#$ -q long.q
#$ -l gpu=1,gpu_ram=32G
#$ -l hostname='supergpu10|supergpu11|supergpu12|supergpu13|supergpu15|supergpu17|supergpu19|supergpu20'
#$ -o /mnt/strade/ilicka/zero-shot-ssr/outputs/diffvc/synth-5/ssr-diffvc-5.$JOB_ID.log
#$ -e /mnt/strade/ilicka/zero-shot-ssr/outputs/diffvc/synth-5/ssr-diffvc-5.$JOB_ID.log

# Before qsub (the -o/-e dir must already exist, SGE won't create it): mkdir -p /mnt/strade/ilicka/zero-shot-ssr/outputs/diffvc/synth-5
source "/mnt/strade/ilicka/zero-shot-ssr/scripts/sge/_sge_header.sh"

SAVE_DIR="/mnt/strade/ilicka/zero-shot-ssr/outputs/diffvc/synth-5"
mkdir -p "$SAVE_DIR"

EMB="embeddings/non-finetuned/organizer-pretrained/exported_embeddings"
# DiffVC (diffusion decoder, frozen LibriTTS avg-voice encoder for content, external g).
# 50 epochs ~= DiffVC's 16.5M-sample budget over the ~327.6k-utt Train-5 subset. gpu_ram=32G:
# batch 32 measured ~22 GB. The frozen-encoder avg-voice mels are cached to libritts_cache/
# (one .npz/subset) and AUTO-POPULATED at startup, before epoch 0. Needs diffvc/checkpts/enc.pt.
# Stage the DiffVC avg-voice mel cache (Train-5) from berta NFS to BeeGFS scratch.
STORAGE_CACHE="$SRCDIR/libritts_cache"
CACHE="/mnt/scratch/tmp/ilicka/zero-shot-ssr/libritts_cache"
mkdir -p "$CACHE"
if [ -f "$STORAGE_CACHE/Train-5.npz" ]; then
    echo "Staging avg-voice mels (Train-5) to scratch at $(date)"
    rsync -a --info=progress2 "$STORAGE_CACHE/Train-5.npz" "$CACHE/Train-5.npz"
fi
# Auto-resume from the newest checkpoint (this run was killed at epoch 34/50, so it picks up
# there, an empty dir just trains from scratch). `sort -V` orders both naming vintages:
# diffvc_ssr_epoch<N>.pth and legacy diffvc_ssr_<samples>.pth. Re-submitting after any kill
# therefore continues rather than restarting.
RESUME=""
CKPT=$(ls -1 "$SAVE_DIR"/diffvc_ssr_*.pth 2>/dev/null | sort -V | tail -1) || true
[ -n "$CKPT" ] && { RESUME="--resume $CKPT"; echo "Resuming from $CKPT"; }

python3 train_diffvc.py \
        --exp-npz "$EMB/train_embeddings.npz" --dev-npz "$EMB/dev_embeddings.npz" --test-npz "$EMB/test_embeddings.npz" \
        --train-synth Train-5 --dev-synth Dev-5 --test-synth Test-5 \
        --slt-root ../shared/datasets/slt-sstc --librispeech-root ../shared/datasets/LibriSpeech \
        --out-dir "$SAVE_DIR" \
        --num-workers 4 --batch-size 32 --epochs 50 \
        --enc-ckpt diffvc/checkpts/enc.pt --libritts-cache "$CACHE" --n-timesteps 6 \
        --wer --asr-model base.en --val-samples 300 --test-samples 0 $RESUME

echo "Syncing avg-voice cache back to storage at $(date)"
mkdir -p "$STORAGE_CACHE"
rsync -a "$CACHE/Train-5.npz" "$STORAGE_CACHE/Train-5.npz"

echo "Job finished at $(date)"
