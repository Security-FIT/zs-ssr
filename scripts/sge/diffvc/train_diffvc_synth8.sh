#!/bin/bash
#$ -N ssr-diffvc-8
#$ -q long.q
#$ -l gpu=1,gpu_ram=32G
#$ -l hostname='supergpu10|supergpu11|supergpu12|supergpu13|supergpu15|supergpu17|supergpu19|supergpu20'
#$ -o /mnt/strade/ilicka/zero-shot-ssr/outputs/diffvc/synth-8/ssr-diffvc-8.$JOB_ID.log
#$ -e /mnt/strade/ilicka/zero-shot-ssr/outputs/diffvc/synth-8/ssr-diffvc-8.$JOB_ID.log

# Before qsub (the -o/-e dir must already exist, SGE won't create it): mkdir -p /mnt/strade/ilicka/zero-shot-ssr/outputs/diffvc/synth-8
source "/mnt/strade/ilicka/zero-shot-ssr/scripts/sge/_sge_header.sh"

SAVE_DIR="/mnt/strade/ilicka/zero-shot-ssr/outputs/diffvc/synth-8"
mkdir -p "$SAVE_DIR"

EMB="embeddings/non-finetuned/organizer-pretrained/exported_embeddings"
# DiffVC (diffusion decoder, frozen LibriTTS avg-voice encoder for content, external g).
# 50 epochs ~= DiffVC's 16.5M-sample budget over the ~327.6k-utt Train-8 subset. gpu_ram=32G:
# batch 32 measured ~22 GB. The frozen-encoder avg-voice mels are cached to libritts_cache/
# (one .npz/subset) and AUTO-POPULATED at startup, before epoch 0. Needs diffvc/checkpts/enc.pt.
# Stage the DiffVC avg-voice mel cache (Train-8) from berta NFS to BeeGFS scratch.
STORAGE_CACHE="$SRCDIR/libritts_cache"
CACHE="/mnt/scratch/tmp/ilicka/zero-shot-ssr/libritts_cache"
mkdir -p "$CACHE"
if [ -f "$STORAGE_CACHE/Train-8.npz" ]; then
    echo "Staging avg-voice mels (Train-8) to scratch at $(date)"
    rsync -a --info=progress2 "$STORAGE_CACHE/Train-8.npz" "$CACHE/Train-8.npz"
fi
python3 train_diffvc.py \
        --exp-npz "$EMB/train_embeddings.npz" --dev-npz "$EMB/dev_embeddings.npz" --test-npz "$EMB/test_embeddings.npz" \
        --train-synth Train-8 --dev-synth Dev-8 --test-synth Test-8 \
        --slt-root ../shared/datasets/slt-sstc --librispeech-root ../shared/datasets/LibriSpeech \
        --out-dir "$SAVE_DIR" \
        --num-workers 4 --batch-size 32 --epochs 50 \
        --enc-ckpt diffvc/checkpts/enc.pt --libritts-cache "$CACHE" --n-timesteps 6 \
        --wer --asr-model base.en --val-samples 300 --test-samples 0
        # --resume "$SAVE_DIR/diffvc_ssr_epoch<N>.pth"

echo "Syncing avg-voice cache back to storage at $(date)"
mkdir -p "$STORAGE_CACHE"
rsync -a "$CACHE/Train-8.npz" "$STORAGE_CACHE/Train-8.npz"

echo "Job finished at $(date)"
