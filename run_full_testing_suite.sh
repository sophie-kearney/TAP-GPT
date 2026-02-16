#!/bin/sh

k=4
p=16
modality="amyloid"

# ALL MODALITIES k=4 p=16

for seed in 36 73 105 254 314 492 564 688 777 825; do
  sbatch  --job-name=${seed}_${modality}_run_tasks_tabularLLM \
          --output=logs/${seed}_${modality}_output_tasks.out \
          --error=logs/${seed}_${modality}_error_tasks.err \
          tasks/run_tasks.sh $seed $k $modality $p
done