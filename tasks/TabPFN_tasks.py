###
# DATA ABSTRACTION
###

import sys
import pandas as pd
import tabpfn
import torch
import random
import numpy as np
import core_functions.data_processing
import core_functions.analysis
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
# LOAD MODEL
###

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = tabpfn.TabPFNClassifier(n_preprocessing_jobs=1, device=device)

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
# FEW SHOT RANDOM k
###

rng = np.random.default_rng(random_state)
results = []

for i in range(len(test_df)):
    # guarantee we get at least one AD example in the ICL examples
    sampled_ad = pool_ad.sample(n=1, random_state=int(rng.integers(0, 2 ** 32 - 1)))
    sampled_rest = test_df_pool.drop(index=sampled_ad.index).sample(n=k - 1,
                                                                    random_state=int(rng.integers(0, 2 ** 32 - 1)))
    sampled_df = pd.concat([sampled_ad, sampled_rest], ignore_index=True).sample(
        frac=1, random_state=int(rng.integers(0, 2 ** 32 - 1))
    )

    y_train = sampled_df['AlzheimersDisease'].values
    x_train = sampled_df.drop(columns=['AlzheimersDisease', 'PTID', 'split'])

    # "test set" is the one unlabeled row
    x_test = test_df.drop(columns=['AlzheimersDisease', 'PTID', 'split']).iloc[[i]]
    y_true = test_df['AlzheimersDisease'].iloc[i]

    if len(np.unique(y_train)) < 2:
        continue

    model.fit(x_train, y_train)

    # predict the unlabeled row
    y_pred_proba = model.predict_proba(x_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    results.append({
        'task': "few_shot_random",
        'model': "TabPFN",
        'ID': test_df['PTID'].iloc[[i]].values[0],
        'AlzheimersDisease': test_df['AlzheimersDisease'].iloc[[i]].values[0],
        'prompt': "NA",
        'y_pred': y_pred[0],
        'y_true': y_true
    })

###
# SAVE RESULTS
###

results_df = pd.DataFrame(results)
results_df.to_csv(f"data/task_results/TabPFN_task_predictions{random_state}_{k}_{modality}.csv", index=False)

###
# GET METRICS
###

results_df = pd.DataFrame(results)

y_true = results_df['y_true']
y_pred = results_df['y_pred']
metrics = core_functions.analysis.get_metrics(y_true, y_pred)
metrics.update({"model": "TabPFN", "task": "few_shot_tabular",
                    "seed": random_state, "k": k, "modality": modality, "p": p})

metrics_df = pd.DataFrame([metrics])
print(metrics_df.to_csv(index=False))
metrics_df.to_csv(f"data/{modality}/TabPFN_{random_state}_metrics_{random_state}.csv", index=False)