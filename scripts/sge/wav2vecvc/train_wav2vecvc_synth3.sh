#!/bin/bash
#$ -N ssr-wav2vecvc-3
#$ -q long.q
#$ -l gpu=1,gpu_ram=20G
#$ -l hostname='supergpu10|supergpu11|supergpu12|supergpu13|supergpu15|supergpu17|supergpu19|supergpu20'
#$ -o /mnt/strade/ilicka/zero-shot-ssr/outputs/wav2vecvc/synth-3/ssr-wav2vecvc-3.$JOB_ID.log
#$ -e /mnt/strade/ilicka/zero-shot-ssr/outputs/wav2vecvc/synth-3/ssr-wav2vecvc-3.$JOB_ID.log

# Before qsub (the -o/-e dir must already exist, SGE won't create it): mkdir -p /mnt/strade/ilicka/zero-shot-ssr/outputs/wav2vecvc/synth-3
source "/mnt/strade/ilicka/zero-shot-ssr/scripts/sge/_sge_header.sh"

SAVE_DIR="/mnt/strade/ilicka/zero-shot-ssr/outputs/wav2vecvc/synth-3"
mkdir -p "$SAVE_DIR"

EMB="embeddings/non-finetuned/organizer-pretrained/exported_embeddings"
# wav2vec-VC: content is a weighted sum of the wav2vec2 layers, extracted every step (no feature
# cache), speaker identity is the external g. The released universal vocoder of wav2vec-VC
# (wav2vecvc/checkpts/vocoder.pt) is autoregressive, which makes the test evaluation slow.
# 20 epochs is about wav2vec-VC's original budget of 6.4M samples.
# The VC wavs are read from one indexed tar per subset (pack_sltsstc.py, skips packed subsets),
# copied to scratch. LibriSpeech targets stay loose files.
SLT_LOOSE="../shared/datasets/slt-sstc"
SLT_TAR="$SLT_LOOSE/tars"
SLT_ROOT="/mnt/scratch/tmp/ilicka/zero-shot-ssr/slt-sstc-tar"
mkdir -p "$SLT_TAR" "$SLT_ROOT"
echo "Packing VC tars (Train-3 Dev-3 Test-3) at $(date)"
python3 pack_sltsstc.py --loose-root "$SLT_LOOSE" --out-root "$SLT_TAR" --synth Train-3 Dev-3 Test-3
for s in Train-3 Dev-3 Test-3; do
    [ -e "$SLT_TAR/$s.index.npz" ] || { echo "ERROR: missing tar index $SLT_TAR/$s.index.npz (pack failed?)" >&2; exit 1; }
    rsync -a --info=progress2 "$SLT_TAR/$s.tar" "$SLT_TAR/$s.index.npz" "$SLT_ROOT/"
done

# continue from the newest checkpoint in SAVE_DIR left by a previous run, if any.
RESUME=""
CKPT=$(ls -t "$SAVE_DIR"/wav2vecvc_ssr_*.pth 2>/dev/null | head -1) || true
[ -n "$CKPT" ] && { RESUME="--resume $CKPT"; echo "Resuming from $CKPT"; }

python3 train_wav2vec-vc.py \
        --exp-npz "$EMB/train_embeddings.npz" --test-npz "$EMB/test_embeddings.npz" \
        --train-synth Train-3 --test-synth Test-3 \
        --slt-root "$SLT_ROOT" --librispeech-root ../shared/datasets/LibriSpeech \
        --out-dir "$SAVE_DIR" \
        --num-workers 4 --batch-size 32 --epochs 20 \
        --test-samples 0 --wer --asr-model base.en $RESUME

echo "Job finished at $(date)"
