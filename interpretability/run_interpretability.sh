#!/bin/sh

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=ai
#SBATCH --mem-per-gpu=160GB

random_state=$1
modality=$2
k=$3
p=$4
task=$5

echo "> START"
source /cbica/software/external/python/anaconda/3/etc/profile.d/conda.sh
echo "> ACTIVATING VENV"
conda activate finetune_env2
echo "> SET UP CUDA"
module unload cuda
module load cuda/11.8

echo "> GENERATING INTERPRETABLE TAPGPT PREDICTIONS"
start_time=$(date +%s)
python interpretability/interpretable_TableGPT.py "$random_state" "$modality" "$k" "$p" "$task"
echo "  $(( $(date +%s) - start_time )) seconds"

echo "> DONE"