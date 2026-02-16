###
# DATA ABSTRACTION
###

import pandas as pd
from datetime import datetime
import os, sys
import core_functions.analysis

pd.set_option('display.max_columns', None)
random_state = int(sys.argv[1])
k = int(sys.argv[2])
p = int(sys.argv[3])
modality = sys.argv[4]

###
# LOAD DATA
###

models = ["TableGPT", "Qwen3", "Qwen2.5-Instruct", "TabPFN", "openai-gpt-4-1-mini","openai-gpt-5", "TableGPT-R1"]
all_results = pd.concat(
    [pd.read_csv(f"data/task_results/{m}_task_predictions{random_state}_{p}_{k}_{modality}.csv") for m in models if os.path.exists(f"data/task_results/{m}_task_predictions{random_state}_{p}_{k}_{modality}.csv")]
)

###
# LOOP THROUGH K
###

evaluation_metrics = []

for m in models:

    curr_mod_res = all_results[all_results['model'] == m]

    tasks = curr_mod_res['task'].unique()

    for t in tasks:
        curr_results = curr_mod_res[curr_mod_res['task'] == t]
        curr_results = curr_results[curr_results['prediction'].isin(["0", "1", 0, 1])]

        y_true = curr_results['AlzheimersDisease'].astype(int)
        y_pred = curr_results['prediction'].astype(int)

        metrics = core_functions.analysis.get_metrics(y_true, y_pred)
        metrics.update({"model": m,"task": t,"seed": random_state, "k": k, "modality": modality})
        evaluation_metrics.append(metrics)

evaluation_metrics_df = pd.DataFrame(evaluation_metrics)

# change order of cols
remaining_cols = [col for col in evaluation_metrics_df.columns if col not in ["model","task","seed","k","modality"]]
evaluation_metrics_df = evaluation_metrics_df[["model","task","seed","k","modality"] + remaining_cols]

print(evaluation_metrics_df.to_csv(None, index=False))
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
evaluation_metrics_df.to_csv(f"data/{modality}/task_metrics{random_state}_{k}_{current_time}.csv", index=False)