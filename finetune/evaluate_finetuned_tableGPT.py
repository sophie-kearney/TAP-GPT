###
# DATA ABSTRACTION
###

import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from transformers import AutoModelForCausalLM, AutoTokenizer
import core_functions.finetune
import core_functions.analysis
import sys, torch
import numpy as np
import random, os

###
# LOAD DATASET
###

task = sys.argv[1] if len(sys.argv) > 1 else "zero_shot_tabular"
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 36
k = int(sys.argv[3]) if len(sys.argv) > 3 else 8
p = int(sys.argv[4]) if len(sys.argv) > 4 else 16
modality = sys.argv[5] if len(sys.argv) > 4 else "amyloid"

READ_TOKEN = os.getenv("READ_TOKEN")
WRITE_TOKEN = os.getenv("WRITE_TOKEN")

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# load data
path = f"data/task_prompts/{task}_{seed}_{modality}_{p}_{k}.csv"
prompts = pd.read_csv(path)
prompts = prompts.rename(columns={
            'prompt': 'input_text',
            'AlzheimersDisease': 'output_text'
})
test_data = prompts[prompts['split'] == 'test'].drop(columns=['split'])

###
# LOAD BASE MODEL
###

BASE_MODEL = "tablegpt/TableGPT2-7B"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    device_map="auto",
    load_in_4bit=True,
    low_cpu_mem_usage=True,
    token=READ_TOKEN
)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    padding_side="left",
    use_fast=False,
    token=READ_TOKEN
)
tokenizer.pad_token = tokenizer.eos_token

###
# GENERATE RESPONSE
###

# TODO - update huggingface user
repo = f"<user>/tableGPT2-{task}-{seed}-{modality}-{p}-{k}"
name = f"{task}_{modality}"

model.load_adapter(repo, adapter_name=name)
model.set_adapter(adapter_name=name)

results = core_functions.finetune.eval_finetuned(model, tokenizer, test_data)

y_pred = results['prediction'].astype(int).values
y_true = results['AlzheimersDisease'].astype(int).values
metrics = core_functions.analysis.get_metrics(y_true, y_pred)
metrics.update({"model": "TAP-GPT", "task": task, "seed": seed, "k": k, "modality": modality, "p":p})

pd.DataFrame([metrics]).to_csv(f"data/{modality}/TAP-GPT_{task}_{seed}_{p}_{k}_metrics.csv", index=False)