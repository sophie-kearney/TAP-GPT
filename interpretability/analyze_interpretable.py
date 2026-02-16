###
# DATA ABSTRACTION
###

import pandas as pd
import os
import core_functions.analysis

pd.set_option('display.max_columns', None)


###
# LOAD DATA
###

tasks = ["few_shot_tabular", "zero_shot_tabular", "zero_shot_tabular", "zero_shot_serialized"]
seeds = [36, 73, 105, 254, 314, 492, 564, 688, 777, 825]
modalities = ["amyloid", "tau", "MRI"]

files = [
    f"data/task_results/TAP-GPT_{task}_{seed}_{modality}_interpretable.csv"
    for task in tasks
    for seed in seeds
    for modality in modalities
    if os.path.exists(f"data/task_results/TAP-GPT_{task}_{seed}_{modality}_interpretable.csv")
]

all_results = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

###
# ANALYZE
###

evaluation_metrics = []

for (t, modality, seed), grp in all_results.groupby(["task", "modality", "seed"]):
    grp = grp[grp["pred_label"].isin(["0", "1", 0, 1])]

    y_true = grp["AlzheimersDisease"].astype(int)
    y_pred = grp["pred_label"].astype(int)

    metrics = core_functions.analysis.get_metrics(y_true, y_pred)
    metrics.update({"model":"Interpretable TAP-GPT", "task": t, "seed": int(seed), "k":4, "modality": modality, "p":16})
    evaluation_metrics.append(metrics)

evaluation_metrics_df = pd.DataFrame(evaluation_metrics)

print(evaluation_metrics_df.to_csv(None, index=False))
