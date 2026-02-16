#!/bin/bash

seeds=(36 73 105 254 314 492 564 688 777 825)
modality="amyloid"
p=16
k=4

# ALL MODALITIES: k=4 p=16

for seed in "${seeds[@]}"; do
    echo "Running traditional ML tasks for $modality seed $seed..."
    /Users/sophiekk/PycharmProjects/finetuneLLM/.venv/bin/python -m tasks.process_data "$seed" "$k" "$p"  "$modality"
    /Users/sophiekk/PycharmProjects/finetuneLLM/.venv/bin/python -m tasks.traditionalML_tasks $seed $k $p $modality
    /Users/sophiekk/PycharmProjects/finetuneLLM/.venv/bin/python -m tasks.TabPFN_tasks $seed $k $p $modality
done