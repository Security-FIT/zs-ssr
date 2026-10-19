#!/bin/bash
#$ -N ssr-utmos
#$ -q long.q
#$ -l gpu=1,gpu_ram=16G
#$ -l hostname='supergpu10|supergpu11|supergpu12|supergpu13|supergpu15|supergpu17|supergpu19|supergpu20'
#$ -o /mnt/strade/ilicka/zero-shot-ssr/outputs/analysis/naturalness/naturalness.$JOB_ID.log
#$ -e /mnt/strade/ilicka/zero-shot-ssr/outputs/analysis/naturalness/naturalness.$JOB_ID.log

# Naturalness of the SLT-SSTC test VC samples (analysis/vc_naturalness.py): UTMOS22 predicted
# MOS (1-5) per VC system. It covers what target similarity misses, e.g. kNN-VC (Test-8) reuses
# real target frames and so has a high similarity, but its stitching gives it a low MOS.
#
# One synthesizer per job, so the 8 test systems run in parallel. The file list comes from
# analysis/results/vc_quality_Test-N.tsv (run the vc_quality jobs first). Each job writes
# analysis/results/naturalness_Test-N.tsv, which plot_main_results.py reads directly.
#
#     for n in 1 2 3 4 5 6 7 8; do qsub -v SYNTH=$n scripts/sge/analysis/vc_naturalness.sh; done
#     qsub -v SYNTH=3,LIMIT=1500 scripts/sge/analysis/vc_naturalness.sh   # random subset, faster
#
# Compute nodes are offline, so UTMOS22 has to be in the shared torch.hub cache. To fill a fresh
# cache, run on the frontend:
#   export TORCH_HOME=/mnt/strade/ilicka/.cache/torch
#   python -c "import torch; torch.hub.load('tarepan/SpeechMOS','utmos22_strong',trust_repo=True)"
#
# Before qsub (the -o/-e dir must already exist, SGE won't create it):
#   mkdir -p /mnt/strade/ilicka/zero-shot-ssr/outputs/analysis/naturalness

source "/mnt/strade/ilicka/zero-shot-ssr/scripts/sge/_sge_header.sh"

# one test synthesizer per job, written to its own naturalness_Test-N.tsv
: "${SYNTH:?pass the test synthesizer number, e.g. qsub -v SYNTH=3 ...}"
echo "naturalness Test-$SYNTH at $(date) (LIMIT=${LIMIT:-0})"
python3 -m analysis.vc_naturalness \
        --slt-root ../shared/datasets/slt-sstc \
        --results-dir "$SRCDIR/analysis/results" \
        --synths "$SYNTH" \
        --limit "${LIMIT:-0}" \
        --out "$SRCDIR/analysis/results/naturalness_Test-$SYNTH.tsv"

echo "Job finished at $(date)"
