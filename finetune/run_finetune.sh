#!/bin/sh

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=ai
#SBATCH --mem-per-gpu=160GB

echo "> START"
source /cbica/software/external/python/anaconda/3/etc/profile.d/conda.sh
echo "> ACTIVATING VENV"
conda activate finetune_env2
echo "> SET UP CUDA"
module unload cuda
module load cuda/11.8

random_state="$1"
k="$2"
task="$3"
modality="$4"
p="$5"

echo "> MAKE PROMPTS, SPLIT DATA"
python -m tasks.process_data "$random_state" "$k" "$p"  "$modality"

echo "> FINETUNE TABLEGPT"
start_time=$(date +%s)
python -m finetune.finetune_TableGPT $task $random_state $k $p $modality
echo "  $(( $(date +%s) - start_time )) seconds"

echo "> EVALUATE FINETUNED MODEL"
start_time=$(date +%s)
python -m finetune.evaluate_finetuned_tableGPT $task $random_state $k $p $modality
echo "  $(( $(date +%s) - start_time )) seconds"

echo "> FINETUNE QWEN2.5"
python -m finetune.finetune_Qwen2_5 $task $random_state $k $p $modality

echo "> EVALUATE FINETUNED QWEN2.5"
start_time=$(date +%s)
python -m finetune.evaluate_finetuned_Qwen2_5 $task $random_state $k $p $modality
echo "  $(( $(date +%s) - start_time )) seconds"

echo "> DONE"