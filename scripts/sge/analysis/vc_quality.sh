#!/bin/bash
#$ -N ssr-vcqual
#$ -q long.q
#$ -l gpu=1,gpu_ram=16G
#$ -l hostname='supergpu10|supergpu11|supergpu12|supergpu13|supergpu15|supergpu17|supergpu19|supergpu20'
#$ -o /mnt/strade/ilicka/zero-shot-ssr/outputs/analysis/vc-quality/vc_quality.$JOB_ID.log
#$ -e /mnt/strade/ilicka/zero-shot-ssr/outputs/analysis/vc-quality/vc_quality.$JOB_ID.log

# Quality of the SLT-SSTC test VC samples (analysis/vc_quality.py), per utterance:
#   * target similarity: ECAPA cosine between the VC wav and one random VoxCeleb utterance of its
#     target speaker, never the clip the conversion was conditioned on
#   * nontarget similarity: the same VC embedding against a random utterance of another VoxCeleb1
#     speaker, giving the per-synthesizer target-speaker EER of the summary line
#   * WER: Whisper transcript of the VC wav against the LibriSpeech source transcript
# One synthesizer (Test-1..16) per job, passed with -v SYNTH:
#     qsub -v SYNTH=Test-1 scripts/sge/analysis/vc_quality.sh
#     ...
#     qsub -v SYNTH=Test-8 scripts/sge/analysis/vc_quality.sh
# The jobs are independent, each ~13530 utterances (~1-2 h, bound by Whisper).
#
# -v SKIP_WER=1 recomputes only the similarity columns (no Whisper, minutes instead of hours) and
# copies the WER columns from the TSV already on disk, e.g. after a change to the trials:
#     for n in $(seq 1 16); do
#         qsub -N ssr-vcqual-$n -v SYNTH=Test-$n,SKIP_WER=1 scripts/sge/analysis/vc_quality.sh
#     done
#
# The VC wavs are read straight from ../shared/datasets/slt-sstc (each is read once, so staging
# to scratch would not pay off). The per-utterance TSV goes to analysis/results/, and the
# `[vc_quality.py] Test-N: ... EER=.. micro_WER=..` line of each job gives the VC EER and WER
# bars of analysis/plot_main_results.py.
#
# Before qsub (the -o/-e dir must already exist, SGE won't create it):
#   mkdir -p /mnt/strade/ilicka/zero-shot-ssr/outputs/analysis/vc-quality

source "/mnt/strade/ilicka/zero-shot-ssr/scripts/sge/_sge_header.sh"

: "${SYNTH:?pass the synthesizer subset, e.g. qsub -v SYNTH=Test-3 ...}"

# Whisper for the WER. Compute nodes have no internet, but base.en only needs to be cached
# once under $HOMEDIR (network-shared), see README "Environment setup". wespeaker (ECAPA)
# is already in the env. Not needed at all under SKIP_WER.
[ "${SKIP_WER:-0}" = "1" ] || python3 -c "import whisper" 2>/dev/null || uv pip install openai-whisper

EMB="embeddings/non-finetuned/organizer-pretrained/exported_embeddings"
OUT="$SRCDIR/analysis/results"
mkdir -p "$OUT"

echo "Analyzing VC quality for $SYNTH at $(date)"
python3 -m analysis.vc_quality \
        --test-npz "$EMB/test_embeddings.npz" \
        --synth "$SYNTH" \
        --slt-root ../shared/datasets/slt-sstc \
        --librispeech-root ../shared/datasets/LibriSpeech \
        --vox-root ../shared/datasets/voxceleb \
        --asr-model base.en \
        ${SKIP_WER:+--skip-wer} \
        --out-dir "$OUT"

echo "Job finished at $(date)"
