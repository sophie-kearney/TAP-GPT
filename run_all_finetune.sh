#!/bin/sh

k=4
p=16
modality="amyloid"
task="few_shot_tabular"
# ALL MODALITIES k=4 p=16

for seed in 36 73 105 254 314 492 564 688 777 825; do
      sbatch  --job-name=${seed}_${modality}_${task}_finetune \
          --output=logs/${seed}_${modality}_${task}_finetune.out \
          --error=logs/${seed}_${modality}_${task}_finetune.err \
          finetune/run_finetune.sh $seed $k $task $modality $p
done