#!/bin/bash
#$ -N ssr-threshold-impact
#$ -q long.q
#$ -o /mnt/strade/ilicka/zero-shot-ssr/outputs/analysis/threshold-impact/ssr-threshold-impact.$JOB_ID.log
#$ -e /mnt/strade/ilicka/zero-shot-ssr/outputs/analysis/threshold-impact/ssr-threshold-impact.$JOB_ID.log

# Threshold impact (analysis/correlate_quality.py): how filtering the VC inputs by their WER,
# duration or target similarity changes the reconstruction EER and WER. CPU only, nothing is
# reconstructed: the EER and WER are recomputed from the test-final_trials_*.tsv of each run and
# the vc_quality_Test-N.tsv files. The slow part is the first run, which reads the duration of
# every VC utterance in those trials (~100k file headers) and caches it to
# analysis/results/duration_cache.tsv, so later runs only measure utterances not seen before.
#
# Defaults: FreeVC, every synthesizer found, 100 thresholds, aligned runs preferred for Test-1/4.
# Override with -v:
#     qsub scripts/sge/analysis/threshold_impact.sh
#     qsub -v ARCHS="freevc diffvc",SYNTHS="1 4",NUM_THRESHOLDS=20 scripts/sge/analysis/threshold_impact.sh
#
# Before qsub (the -o/-e dir must already exist, SGE won't create it):
#   mkdir -p /mnt/strade/ilicka/zero-shot-ssr/outputs/analysis/threshold-impact

NGPUS=0   # CPU-only job: skip the header's GPU selection
source "/mnt/strade/ilicka/zero-shot-ssr/scripts/sge/_sge_header.sh"

ARCHS_ARG=""
[ -n "${ARCHS:-}" ] && ARCHS_ARG="--archs $ARCHS"
SYNTHS_ARG=""
[ -n "${SYNTHS:-}" ] && SYNTHS_ARG="--synths $SYNTHS"

echo "Running threshold-impact analysis at $(date)"
python3 -m analysis.correlate_quality \
        --results-dir analysis/results --outputs-dir outputs \
        --slt-root ../shared/datasets/slt-sstc \
        --num-thresholds "${NUM_THRESHOLDS:-100}" \
        $ARCHS_ARG $SYNTHS_ARG \
        --out analysis/figures/threshold_impact.svg \
        --prefer-align ${PREFER_ALIGN:-1 4}

echo "Job finished at $(date)"
