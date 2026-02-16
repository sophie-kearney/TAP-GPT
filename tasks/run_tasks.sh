#!/bin/sh

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=ai
#SBATCH --mem-per-gpu=160GB

random_state=$1
k=$2
modality=$3
p=$4

echo "> START"
cd /gpfs/fs001/cbica/comp_space/kearnes/tabLLM_imaging
echo "> CWD: $(pwd)"
source /cbica/software/external/python/anaconda/3/etc/profile.d/conda.sh
echo "> ACTIVATING VENV"
conda activate llm_env
echo "> SET UP CUDA"
module unload cuda
module load cuda/11.8

echo "> MAKE PROMPTS, SPLIT DATA"
python -m tasks.process_data "$random_state" "$k" "$p"  "$modality"

echo "> GENERATING VANILLA LLMs"
start_time=$(date +%s)
python -m tasks.vanilla_llms_tasks "$random_state" "$k" "$p" "$modality"
echo "  $(( $(date +%s) - start_time )) seconds"

echo "> GENERATING TABPFN PREDICTIONS"
start_time=$(date +%s)
python -m tasks.TabPFN_tasks "$random_state" "$k" "$p" "$modality"
echo "  $(( $(date +%s) - start_time )) seconds"

echo "> ANALYZING RESULTS"
python  -m tasks.analyze_task_metrics "$random_state" "$k" "$p" "$modality"

echo "> DONE"