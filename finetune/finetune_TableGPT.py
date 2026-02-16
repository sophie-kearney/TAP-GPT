###
# DATA ABSTRACTION
###

import core_functions.finetune
import pandas as pd
import sys, os
from datasets import Dataset

task = sys.argv[1]
seed = int(sys.argv[2])
k = int(sys.argv[3])
p = int(sys.argv[4])
modality = sys.argv[5]

if len(sys.argv) > 6:
    scisub9 = bool(int(sys.argv[6]))
else:
    scisub9 = False

READ_TOKEN = os.getenv("READ_TOKEN")
WRITE_TOKEN = os.getenv("WRITE_TOKEN")

###
# CHECK IF MODEL EXISTS
###

if scisub9:
    new_model_name = f"tableGPT2-{task}-{seed}-{modality}-{p}-{k}-scisub9"
else:
    new_model_name = f"tableGPT2-{task}-{seed}-{modality}-{p}-{k}"

base_model_name = "tablegpt/TableGPT2-7B"

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