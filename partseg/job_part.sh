#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=64G
#SBATCH --partition=boost_usr_prod
#SBATCH --account=aifpt_agrifood
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
# usage: sbatch -J part_cmp --export=ALL,ARGS="--classchoice all --model v2" partseg/job_part.sh
set -euo pipefail
source /leonardo_scratch/fast/AIFPT_agrifood/miniforge3/etc/profile.d/conda.sh; conda activate geoze
cd /leonardo_scratch/fast/AIFPT_agrifood/code/GeoZe/partseg; mkdir -p ../logs out
eval "python ${SCRIPT:-part_run.py} ${ARGS:-}"
