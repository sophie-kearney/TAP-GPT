###
# DATA ABSTRACTION
###
import sys, random, os
import core_functions.models, core_functions.analysis
import pandas as pd
import numpy as np

random_state = int(sys.argv[1])
k = int(sys.argv[2])
p = int(sys.argv[3])
modality = sys.argv[4]

random.seed(random_state)
np.random.seed(random_state)

###
# LOAD DATA
###

tasks = ["zero_shot_tabular", "few_shot_tabular", "zero_shot_serialized", "few_shot_serialized",]
dfs = []

for task in tasks:
    path = f"data/task_prompts/{task}_{random_state}_{modality}_{p}_{k}.csv"
    # print(path)
    if os.path.exists(path):
        df_task = pd.read_csv(path)
        df_task["task"] = task
        dfs.append(df_task)
if len(dfs) == 0:
    sys.exit("No task prompt files found. Exiting.")

all_prompts = pd.concat(dfs, ignore_index=True)
all_prompts = all_prompts[all_prompts["split"] == "test"]

###
# RUN HUGGINGFACE MODELS
###

models = {"Qwen2.5-Instruct": "Qwen/Qwen2.5-7B-Instruct",
          "Qwen3": "Qwen/Qwen3-8B",
          "TableGPT": "tablegpt/TableGPT2-7B",
          "TableGPT-R1": "tablegpt/TableGPT-R1"}

for m in models.keys():
    print(f"   RUNNING {m}")
    # get predictions
    results = core_functions.models.load_huggingface_model(models[m], m, all_prompts)
    results.to_csv(f"data/task_results/{m}_task_predictions{random_state}_{p}_{k}_{modality}.csv", index=False)

###
# RUN OPENAI MODELS
###

models = {"openai-gpt-4-1-mini":"openai-gpt-4-1-mini"}
DATABRICKS_TOKEN = os.environ.get('DATABRICKS_TOKEN')
API_URL = os.environ.get('API_URL')

for m in models.keys():
    print(f"   RUNNING {m}")
    results = core_functions.models.load_openai_model(models[m], m, DATABRICKS_TOKEN, API_URL, all_prompts)
    results.to_csv(f"data/task_results/{m}_task_predictions{random_state}_{p}_{k}_{modality}.csv", index=False)