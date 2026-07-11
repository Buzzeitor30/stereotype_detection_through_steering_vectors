#!/bin/bash
#SBATCH --job-name=steering_pca
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=12
#SBATCH --time=01:00:00

# Test of the PCA method: trains a steering vector on the Spanish
# stereotype subset using contrastive persona prompts (tweet mode).
#
# Usage:
#   sbatch scripts/train_steering_vector_pca.slurm
#   MODEL_NAME=Qwen/Qwen2.5-7B-Instruct sbatch scripts/train_steering_vector_pca.slurm
#   LAYER_IDS="10 11 12" sbatch scripts/train_steering_vector_pca.slurm
#
# NOTE: logs/ must exist before submitting, or SLURM cannot write the
# output files and the job dies silently.

set -euo pipefail

# Uncomment/adjust for your cluster's environment:
# module load cuda

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p vectors

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
LAYER_IDS="${LAYER_IDS:-}"          # empty = all steerable layers (1..N-1)
CSV_PATH="${CSV_PATH:-../data/stereotype_subset_200_es.csv}"
OUT_VECTOR="${OUT_VECTOR:-vectors/stereotype_es_pca_${SLURM_JOB_ID:-local}.gguf}"

echo "Job ${SLURM_JOB_ID:-<none>} on $(hostname): model=$MODEL_NAME layers=${LAYER_IDS:-all} csv=$CSV_PATH"

uv run python train_steering_vector.py "$CSV_PATH" \
    --model-name "$MODEL_NAME" \
    --tweet-col tweet \
    --method pca \
    --save-vector "$OUT_VECTOR" \
    ${LAYER_IDS:+--layer-ids $LAYER_IDS}

echo "Done. Vector written to $OUT_VECTOR"
