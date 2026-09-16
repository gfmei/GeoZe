#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=64G
#SBATCH --partition=boost_usr_prod
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
# usage: sbatch -J part --export=ALL,ARGS="--classchoice all" partseg/job_part.sh
#        sbatch -J part --export=ALL,SCRIPT=part_run.py,ARGS="--model geoze" partseg/job_part.sh
set -euo pipefail
source /leonardo_scratch/fast/AIFPT_agrifood/miniforge3/etc/profile.d/conda.sh; conda activate geoze
cd /leonardo_scratch/fast/AIFPT_agrifood/code/GeoZe/partseg; mkdir -p ../logs out
eval "python ${SCRIPT:-simple_geoze.py} ${ARGS:-}"
