###
# DATA ABSTRACTION
###

import core_functions.finetune
import pandas as pd
import sys, os
from datasets import Dataset

READ_TOKEN = os.getenv("READ_TOKEN")
WRITE_TOKEN = os.getenv("WRITE_TOKEN")

task = sys.argv[1]
seed = int(sys.argv[2])
k = int(sys.argv[3])
p = int(sys.argv[4])
modality = sys.argv[5]

###
# CHECK IF MODEL EXISTS
###

new_model_name = f"qwen2.5-{task}-{seed}-{modality}-{p}-{k}"
base_model_name = "Qwen/Qwen2.5-7B-Instruct"

exists = core_functions.finetune.model_exists(new_model_name)
if exists:
    exit(0)

###
# LOAD DATA
###

# load data
path = f"data/task_prompts/{task}_{seed}_{modality}_{p}_{k}.csv"
prompts = pd.read_csv(path)

# process data
prompts = prompts.rename(columns={
    'prompt': 'input_text',
    'AlzheimersDisease': 'output_text'
})
train_data = prompts[prompts['split'] == 'train'].drop(columns=['split'])
test_data = prompts[prompts['split'] == 'test'].drop(columns=['split'])
train_data, test_data = Dataset.from_pandas(train_data), Dataset.from_pandas(test_data)

###
# FINETUNE
###

# train model
train_result, eval_result = core_functions.finetune.finetune_huggingface(READ_TOKEN, WRITE_TOKEN,
                                                                         train_data, test_data,
                                                                         new_model_name, base_model_name, seed)
print(train_result)
print(eval_result)