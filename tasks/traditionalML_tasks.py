###
# DATA ABSTRACTION
###

import pandas as pd
import random
import numpy as np
import core_functions.data_processing
import core_functions.tradML
import core_functions.analysis
import sys
pd.set_option('display.max_columns', None)

###
# CONSTANTS
###

random_state = int(sys.argv[1])
k = int(sys.argv[2])
p = int(sys.argv[3])
modality = sys.argv[4]

random.seed(random_state)
np.random.seed(random_state)

###
# LOAD DATA
###

df = pd.read_csv(f"data/data_splits/{random_state}_{modality}_datasplit_{p}_{k}.csv")

if p != 0:
    cols_to_exclude = ["PTID", "split"]
    covariates = ["APOE4", "Gender", "Education", "Age"]
    selected_features = core_functions.data_processing.do_feature_selection(df, p,
                                                                            [*cols_to_exclude, *covariates])
else:
    selected_features = df.columns.tolist()
df = df[selected_features]

# split data
test_df = df[df['split'] == "test"].fillna(0)
test_df_pool = df[df['split'] == "test_pool"].fillna(0)
pool_ad = test_df_pool[test_df_pool["AlzheimersDisease"] == 1]

###
# RUN MODEL
###

predictions = core_functions.tradML.run_traditional_ML_models(df, random_state, k, p, modality)
all_metrics = []

for model_name, data in predictions.items():
    y_true = data['y_true']
    y_pred = data['y_pred']
    metrics = core_functions.analysis.get_metrics(y_true, y_pred)
    metrics.update({"model": model_name, "task": "few_shot_tabular",
                    "seed": random_state, "k": k, "modality": modality, "p": p})
    all_metrics.append(metrics)

metrics_df = pd.DataFrame(all_metrics)
print(metrics_df.to_csv(index=False))
metrics_df.to_csv(f"data/{modality}/traditionalML_metrics{random_state}_{p}_{k}.csv", index=False)