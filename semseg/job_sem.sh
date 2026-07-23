#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 --mem=120G
#SBATCH --partition=boost_usr_prod
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
source /leonardo_scratch/fast/AIFPT_agrifood/miniforge3/etc/profile.d/conda.sh; conda activate spconv
export HF_HOME="/leonardo_scratch/fast/AIFPT_agrifood/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /leonardo_scratch/fast/AIFPT_agrifood/code/GeoZe; mkdir -p logs semseg/out
eval "python semseg/sem_run.py ${ARGS:-}"
