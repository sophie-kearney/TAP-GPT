#!/bin/sh

k=4
p=16
modality="amyloid"
task="zero_shot_tabular"
# ALL MODALITIES k=4 p=16

for seed in 36 73 105 254 314 492 564 688 777 825; do

  sbatch --job-name=${seed}_${modality}tablegpt_interpretability \
          --output=logs/${seed}_${modality}tablegpt_interpretability.out \
          --error=logs/${seed}_${modality}tablegpt_interpretability.err \
          interpretability/run_interpretability.sh $seed $modality $k $p $task
done