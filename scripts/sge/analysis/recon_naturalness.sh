#!/bin/bash
#$ -N ssr-recon-utmos
#$ -q long.q
#$ -l gpu=1,gpu_ram=16G
#$ -l hostname='supergpu10|supergpu11|supergpu12|supergpu13|supergpu15|supergpu17|supergpu19|supergpu20'
#$ -o /mnt/strade/ilicka/zero-shot-ssr/outputs/analysis/recon-naturalness/recon_naturalness.$JOB_ID.log
#$ -e /mnt/strade/ilicka/zero-shot-ssr/outputs/analysis/recon-naturalness/recon_naturalness.$JOB_ID.log

# Naturalness of the reconstructed speech (analysis/recon_naturalness.py): UTMOS22 predicted MOS
# (1-5) of a trained reconstructor's output, to compare with the VC input scored by
# vc_naturalness.sh. Does the reconstructor produce natural speech, and how much does it gain or
# lose against its input?
#
# The reconstructions are not stored, so they are synthesized again, which is the main cost
# (DiffVC runs a diffusion sampler). One (arch, run) per job, passed with -v:
#     qsub -v ARCH=freevc,SYNTH=2 scripts/sge/analysis/recon_naturalness.sh
#     qsub -v ARCH=freevc,SYNTH=1,RUN=synth-1-align scripts/sge/analysis/recon_naturalness.sh
#     qsub -v ARCH=diffvc,SYNTH=4,RUN=synth-4-align,LIMIT=300 scripts/sge/analysis/recon_naturalness.sh
# RUN defaults to synth-<SYNTH>, set it to the -align directory for aligned runs (the test subset
# stays Test-<SYNTH>). LIMIT=0 (default) scores the whole subset (~13.5k utterances), the same
# utterances as the VC input, so the means are comparable. Each job writes
# analysis/results/recon_naturalness_<arch>_<run>.tsv, which plot_main_results.py reads directly.
#
# UTMOS22 must be in the shared torch.hub cache (see vc_naturalness.sh). The reconstructor's own
# pretrained parts (WavLM, DiffVC encoder, vocoders) come from the caches set up by the header.
#
# Before qsub (the -o/-e dir must already exist, SGE won't create it):
#   mkdir -p /mnt/strade/ilicka/zero-shot-ssr/outputs/analysis/recon-naturalness

source "/mnt/strade/ilicka/zero-shot-ssr/scripts/sge/_sge_header.sh"

: "${ARCH:=freevc}"
: "${SYNTH:?pass the test synthesizer number, e.g. qsub -v ARCH=freevc,SYNTH=2 ...}"
: "${SPLIT:=test}"
: "${LIMIT:=0}"   # 0 = the whole subset
OUTDIR="outputs/$ARCH/${RUN:-synth-$SYNTH}"

echo "Recon UTMOS: $ARCH $OUTDIR (Test-$SYNTH, $SPLIT), LIMIT=$LIMIT at $(date)"
python3 -m analysis.recon_naturalness \
        --arch "$ARCH" \
        --synth "$SYNTH" \
        --split "$SPLIT" \
        --out-dir "$OUTDIR" \
        --limit "$LIMIT" \
        --emb-dir embeddings/non-finetuned/organizer-pretrained/exported_embeddings \
        --slt-root ../shared/datasets/slt-sstc \
        --librispeech-root ../shared/datasets/LibriSpeech

echo "Job finished at $(date)"
