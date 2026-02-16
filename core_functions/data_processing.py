###
# DATA ABSTRACTION
###

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import numpy as np
from math import floor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pd.set_option('display.max_columns', None)

###
# SPLIT DATA
###

def split_data(random_state, k, p, modality):
    random.seed(random_state)
    np.random.seed(random_state)

    # ---- LOAD DATA ----
    if modality == "tau":
        df = pd.read_csv("data/tau_DX_multi_visit_subcortical.csv")
    elif modality == "MRI":
        df = pd.read_csv("data/MRI_DX_multi_visit_subcortical.csv")
    else:
        df = pd.read_csv("data/amyloid_DX_multi_visit_subcortical.csv")

    # --- DATA PROCESSING ----
    # get only last patient
    df = df[df["DIAGNOSIS"].isin([1, 3])]
    df = df.sort_values("SCANDATE").groupby("PTID").tail(1)

    if modality == "MRI":
        df = df.filter(regex="_VOLUME").join(df[["PTID", "DIAGNOSIS"]])
    else:
        df = df.filter(regex="_SUVR").join(df[["PTID", "DIAGNOSIS"]])

    # add covariates
    adni_merge = pd.read_csv("data/ADNIMERGE_25Aug2025.csv", low_memory=False)
    adni_merge = adni_merge[["PTID", "APOE4", "AGE", "PTGENDER", "PTEDUCAT"]].drop_duplicates(subset="PTID",
                                                                                              keep="first")
    df = df.merge(adni_merge, on="PTID", how="left")

    # refactor column names and values
    df = df[[col for col in df.columns if col != 'DIAGNOSIS'] + ['DIAGNOSIS']]
    df = df.rename(columns={"DIAGNOSIS": "AlzheimersDisease"})
    df['AlzheimersDisease'] = df['AlzheimersDisease'].replace({1: 0, 3: 1}).astype(int)
    df["PTGENDER"] = df["PTGENDER"].map({"Male": 0, "Female": 1}).astype("Int64")
    df = df.rename(columns={"PTGENDER": "Gender", "PTEDUCAT": "Education", "AGE": "Age"})

    # --- SPLIT DATA ---
    # get test set first - 20% of the data
    df_tmp, test_set = train_test_split(df,
                                        test_size=0.2,
                                        stratify=df['AlzheimersDisease'],
                                        random_state=random_state)
    test_set['split'] = 'test'

    # get exact sizes for remaining splits - split the rest into these sets: 10% validation, 37% train/finetune, 33% k pools
    n = len(df)
    n_val = floor(0.10 * n)
    n_train = floor(0.37 * n)
    n_pool_each = floor(0.11 * n)  # 11% x 3 = 33%
    used = len(test_set) + n_val + n_train + 3 * n_pool_each
    remainder = n - used
    n_train += remainder  # assign any leftovers to train set

    # split off validation and training sets
    df_tmp, val_set = train_test_split(df_tmp,
                                       test_size=n_val,
                                       stratify=df_tmp['AlzheimersDisease'],
                                       random_state=42)
    val_set['split'] = 'val'

    train_set, df_tmp = train_test_split(df_tmp,
                                         train_size=n_train,
                                         stratify=df_tmp['AlzheimersDisease'],
                                         random_state=random_state)
    train_set['split'] = 'train'

    # split off k pools - 3 sets of 11% each
    k_pool_test, df_tmp = train_test_split(df_tmp,
                                           train_size=n_pool_each,
                                           stratify=df_tmp['AlzheimersDisease'],
                                           random_state=random_state)
    k_pool_test['split'] = 'test_pool'
    k_pool_train, df_tmp = train_test_split(df_tmp,
                                            train_size=n_pool_each,
                                            stratify=df_tmp['AlzheimersDisease'],
                                            random_state=random_state)
    k_pool_train['split'] = 'train_pool'
    k_pool_val = df_tmp
    k_pool_val['split'] = 'val_pool'

    combined_df = pd.concat([test_set, val_set, train_set, k_pool_test, k_pool_train, k_pool_val])
    return combined_df

def merge_modalities_split(random_state):
    random.seed(random_state)
    np.random.seed(random_state)

    # --- LOAD DATA ---
    tau_og = pd.read_csv("data/tau_DX_multi_visit_subcortical.csv")
    tau_og = tau_og[[c for c in tau_og.columns if not c.endswith("_VOLUME")]]
    amy = pd.read_csv("data/amyloid_DX_multi_visit_subcortical.csv")
    amy = amy[[c for c in amy.columns if not c.endswith("_VOLUME")]]
    mri_og = pd.read_csv("data/MRI_DX_multi_visit_subcortical.csv")

    tau_og["SCANDATE"] = pd.to_datetime(tau_og["SCANDATE"])
    amy["SCANDATE"] = pd.to_datetime(amy["SCANDATE"])
    mri_og["SCANDATE"] = pd.to_datetime(mri_og["SCANDATE"])

    tau = tau_og.drop(columns=["DIAGNOSIS", "VISCODE"])
    mri = mri_og.drop(columns=["DIAGNOSIS", "VISCODE"])

    tau = tau.rename(columns=lambda c: f"{c}_tau" if c not in ["PTID"] else c)
    amy = amy.rename(columns=lambda c: f"{c}_amy" if c not in ["PTID", "VISCODE", "DIAGNOSIS"] else c)
    mri = mri.rename(columns=lambda c: f"{c}_mri" if c not in ["PTID"] else c)

    # --- MERGE AMY AND TAU ---

    common_ptids = np.intersect1d(amy["PTID"].unique(), tau["PTID"].unique())

    amy_c = amy.loc[amy["PTID"].isin(common_ptids), ["PTID", "SCANDATE_amy"] + [c for c in amy.columns if
                                                                                c not in ["PTID",
                                                                                          "SCANDATE_amy"]]].copy()
    tau_c = tau.loc[tau["PTID"].isin(common_ptids), ["PTID", "SCANDATE_tau"] + [c for c in tau.columns if
                                                                                c not in ["PTID",
                                                                                          "SCANDATE_tau"]]].copy()

    amy_c = amy_c.dropna(subset=["SCANDATE_amy"]).sort_values(["PTID", "SCANDATE_amy"], kind="mergesort").reset_index(
        drop=True)
    tau_c = tau_c.dropna(subset=["SCANDATE_tau"]).sort_values(["PTID", "SCANDATE_tau"], kind="mergesort").reset_index(
        drop=True)

    matched = pd.concat(
        [
            pd.merge_asof(
                g_amy.sort_values("SCANDATE_amy"),
                g_tau.sort_values("SCANDATE_tau"),
                left_on="SCANDATE_amy",
                right_on="SCANDATE_tau",
                direction="nearest",
                tolerance=pd.Timedelta(days=365)
            ).assign(PTID=ptid)
            for ptid, g_amy in amy_c.groupby("PTID", sort=False)
            for g_tau in [tau_c[tau_c["PTID"] == ptid]]
            if len(g_tau)
        ],
        ignore_index=True)

    # --- CHOOSE THE LATEST AMY-TAU PAIR ---
    # merge function above will put an empty tau next to an MRI that doesn't have a match, we need to remove those rows
    matched = matched[matched["SCANDATE_tau"].notna()].copy()
    # get the diff in time between scans
    matched["dt_days"] = (matched["SCANDATE_amy"] - matched["SCANDATE_tau"]).abs().dt.days

    best = (
        matched.sort_values(["PTID", "SCANDATE_amy"], kind="mergesort")
        .drop_duplicates("PTID", keep="last")
        .reset_index(drop=True)
    )

    best = best.drop(columns=[c for c in ["PTID_x"] if c in best.columns])

    # --- MERGE IN MRI ---
    # remove duplicate columns
    best = best.drop(columns=[c for c in best.columns if c.endswith("_y")])
    best = best.rename(columns={c: c[:-2] for c in best.columns if c.endswith("_x")})

    cohort = best.merge(
        mri,
        left_on=["PTID", "SCANDATE_amy"],
        right_on=["PTID", "SCANDATE_mri"],
        how="left"
    )

    # --- FORMAT DATA ---

    cohort["DIAGNOSIS"] = cohort["DIAGNOSIS"].map({1: int(0), 3: int(1)})
    # remove patients that are MCI
    cohort = cohort.dropna(subset=["DIAGNOSIS"])
    cohort = cohort.rename(columns={"DIAGNOSIS": "AlzheimersDisease"})

    # --- GET SHARED TEST SET AND TEST POOL ---
    test_set, rest_pool = train_test_split(
        cohort,
        test_size=0.75,
        stratify=cohort["AlzheimersDisease"],
        random_state=random_state,
    )
    test_pool, rest = train_test_split(
        rest_pool,
        test_size=0.867,
        stratify=rest_pool["AlzheimersDisease"],
        random_state=random_state,
    )

    test_set["split"] = "test"
    test_pool["split"] = "test_pool"
    rest["split"] = "unassigned"

    split_df = pd.concat([test_set, test_pool])

    # --- PIVOT LONGER ---
    id_cols = ["PTID", "AlzheimersDisease", "VISCODE", "split"]

    roi_cols = [c for c in split_df.columns if c.endswith(("_SUVR_tau", "_SUVR_amy", "_VOLUME_mri"))]

    long = split_df[id_cols + roi_cols].melt(
        id_vars=id_cols,
        value_vars=roi_cols,
        var_name="feature",
        value_name="value"
    )

    long["modality"] = long["feature"].str.extract(r"(tau|amy|mri)$")
    long["modality"] = long["modality"].map({"amy": "amyloid", "tau": "tau", "mri": "MRI"})
    long["feature"] = (
        long["feature"]
        .str.replace("_SUVR_tau", "", regex=False)
        .str.replace("_SUVR_amy", "", regex=False)
        .str.replace("_VOLUME_mri", "", regex=False)
    )

    # --- RECONSTRUCT TABLE, ONE ROW PER (PATIENT, MODALITY) ---

    final = (
        long.pivot_table(
            index=["PTID", "AlzheimersDisease", "VISCODE","modality", "split"],
            columns="feature",
            values="value"
        )
        .reset_index()
    )

    # --- ADD IN ADDITIONAL AMYLOID AND MRI PATIENTS FOR TRAINING ---
    test_pool_ptids = final["PTID"].unique()

    # get patients in each modality not in test set or test pool
    amy_last_pat = amy[
        (amy["DIAGNOSIS"].isin([1, 3])) &
        (~amy["PTID"].isin(test_pool_ptids))
        ].sort_values("SCANDATE_amy").groupby("PTID").tail(1)
    tau_last_pat = tau_og[
        (tau_og["DIAGNOSIS"].isin([1, 3])) &
        (~tau_og["PTID"].isin(test_pool_ptids))
        ].sort_values("SCANDATE").groupby("PTID").tail(1)
    mri_last_pat = mri_og[
        (mri_og["DIAGNOSIS"].isin([1, 3])) &
        (~mri_og["PTID"].isin(test_pool_ptids))
        ].sort_values("SCANDATE").groupby("PTID").tail(1)

    # add modality column
    amy_last_pat["modality"] = "amyloid"
    tau_last_pat["modality"] = "tau"
    mri_last_pat["modality"] = "MRI"

    # get train, train_pool, val, and val_pool for each modality
    for df in [amy_last_pat, tau_last_pat, mri_last_pat]:
        df["DIAGNOSIS"] = df["DIAGNOSIS"].map({1: int(0), 3: int(1)})
        df = df.rename(columns={"DIAGNOSIS": "AlzheimersDisease"})
        df = df.rename(
            columns=lambda c: (
                c.replace("_VOLUME", "")
                .replace("_SUVR", "")
                .replace("_amy", "")
                .replace("_tau", "")
                .replace("_mri", "")
            )
        )
        df.drop(columns=["SCANDATE"], inplace=True)

        train, remainder = train_test_split(
            df,
            train_size=0.60,
            stratify=df["AlzheimersDisease"],
            random_state=random_state
        )
        train["split"] = "train"

        val, pools = train_test_split(
            remainder,
            train_size=0.50,
            stratify=remainder["AlzheimersDisease"],
            random_state=random_state
        )
        val["split"] = "val"

        train_pool, val_pool = train_test_split(
            pools,
            train_size=0.50,
            stratify=pools["AlzheimersDisease"],
            random_state=random_state
        )
        train_pool["split"] = "train_pool"
        val_pool["split"] = "val_pool"


        final = pd.concat([final, train, val, train_pool, val_pool])

    # print(pd.crosstab(final["modality"], final["split"]))

    # --- ADD COVARIATES ---
    # add covariates
    adni_merge = pd.read_csv("data/ADNIMERGE_25Aug2025.csv", low_memory=False)
    adni_merge = (
        adni_merge[["PTID", "APOE4", "AGE", "PTGENDER", "PTEDUCAT"]]
        .drop_duplicates(subset="PTID", keep="first")
    )
    final = final.merge(adni_merge, on="PTID", how="left")
    final["PTGENDER"] = final["PTGENDER"].map({"Male": 0, "Female": 1}).astype("Int64")
    final = final.rename(columns={"PTGENDER": "Gender", "PTEDUCAT": "Education", "AGE": "Age"})

    return final

def do_feature_selection(df, p, cols_to_exclude):
    # define dataset
    train_set = df[df["split"] == "train"]
    X = train_set.drop(columns=["AlzheimersDisease", *cols_to_exclude])
    X.fillna(0, inplace=True)
    y = train_set["AlzheimersDisease"].astype(int)

    # define model
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="liblinear", max_iter=5000))
    ])

    # get top features
    model.fit(X, y)
    coef = model.named_steps["clf"].coef_.ravel()
    top_p = (
        pd.Series(np.abs(coef), index=X.columns)
        .sort_values(ascending=False)
        .head(p)
        .index
        .tolist()
    )
    return [*cols_to_exclude, *top_p, "AlzheimersDisease"]

###
# CREATE TASK PROMPTS
###

def create_zero_shot_tabular(instruction_header, zero_shot_instruction, df,cols_to_exclude):
    df_noad = df.drop(columns=cols_to_exclude + ["AlzheimersDisease"])
    prompts = df_noad.apply(
        lambda row: f"""{instruction_header}
### Instruction: {zero_shot_instruction}
### Input:\n{row.to_frame().T.to_string(index=False)}
### Response: """,
        axis=1
    )
    new_df = pd.DataFrame({'ID': df['PTID'],
                           'prompt': prompts,
                           'AlzheimersDisease': df['AlzheimersDisease'],
                           'split': df['split']})
    return new_df

def create_few_shot_tabular(instruction_header, few_shot_instruction, df, random_state, k, cols_to_exclude):
    rng = np.random.default_rng(random_state)
    for split in ['train', 'val', 'test']:
        # sample k examples from the pool for the few-shot examples
        pool = df[df['split'] == f"{split}_pool"]
        pool_ad = pool[pool["AlzheimersDisease"] == 1]
        curr_df = df[df['split'] == split]

        # copy df to hide the labels
        hidden_df = curr_df.copy()
        hidden_df['AlzheimersDisease'] = 'X'
        hidden_df = hidden_df.drop(columns=cols_to_exclude)

        prompts = []
        for i in range(len(hidden_df)):
            # guarantee we get at least one AD example in the ICL examples
            sampled_ad = pool_ad.sample(n=1, random_state=int(rng.integers(0, 2 ** 32 - 1)))
            sampled_rest = pool.drop(index=sampled_ad.index).sample(n=k - 1, random_state=int(rng.integers(0, 2 ** 32 - 1)))
            sampled_df = pd.concat([sampled_ad, sampled_rest], ignore_index=True).sample(
                frac=1, random_state=int(rng.integers(0, 2 ** 32 - 1))
            )

            # add the current row to the end of the static few-shot examples
            combined = pd.concat([sampled_df, hidden_df.iloc[[i]]])
            combined.drop(columns=cols_to_exclude, inplace=True)

            table = combined.head(k + 1).to_string(index=False)

            # make tabular promtp
            prompt = f"""{instruction_header}
### Instruction: {few_shot_instruction}
### Input:\n{table}
### Response: """
            prompts.append(prompt)

        # create a new dataframe with the prompts and labels
        if split == 'train':
            all_df_tabular = pd.DataFrame({'ID': curr_df['PTID'],
                                           'prompt': prompts,
                                           'AlzheimersDisease': curr_df['AlzheimersDisease'],
                                           'split': split})
        else:
            all_df_tabular = pd.concat([all_df_tabular, pd.DataFrame({'ID': curr_df['PTID'],
                                                                      'prompt': prompts,
                                                                      'AlzheimersDisease': curr_df['AlzheimersDisease'],
                                                                      'split': split})])
    return all_df_tabular

def create_zero_shot_tabular_spatial(instruction_header, zero_shot_instruction, df, rois):
    rows = []
    for id in df["PTID"].unique():
        curr = df[df["PTID"] == id]

        prompt = f"""{instruction_header}
    ### Instruction: {zero_shot_instruction}
    ### Input:\n{curr[rois].to_string(index=False)}
    ### Response: """

        rows.append({'ID': id,
                     'prompt': prompt,
                     'AlzheimersDisease': int(curr['AlzheimersDisease'].iloc[0]),
                     'split': curr['split'].iloc[0],
                     })
    return(pd.DataFrame(rows))

def create_few_shot_tabular_spatial(instruction_header, few_shot_instruction, df, rois, random_state, k):
    rng = np.random.default_rng(random_state)
    out_rows = []
    for split in ['train', 'test']:
        # define pool for sampling
        pool_df = df[df["split"] == f"{split}_pool"]
        pool_ptids = pool_df["PTID"].unique()
        pool_ad_ptids = pool_df.loc[pool_df["AlzheimersDisease"] == 1, "PTID"].unique()
        curr_df = df[df['split'] == split]

        # copy df to hide the labels
        hidden_df = curr_df.copy()
        hidden_df['AlzheimersDisease'] = 'X'

        prompts = []
        for ptid in curr_df["PTID"].unique():
            # guarantee we get at least one AD example in the ICL examples
            ad_ptid = rng.choice(pool_ad_ptids, size=1, replace=False)[0]
            rest_candidates = pool_ptids[pool_ptids != ad_ptid]
            rest_ptids = rng.choice(rest_candidates, size=k - 1, replace=False)
            sampled_ptids = np.concatenate([[ad_ptid], rest_ptids])
            rng.shuffle(sampled_ptids)

            sampled_df = pool_df[pool_df["PTID"].isin(sampled_ptids)]
            curr_patient_hidden = hidden_df[hidden_df["PTID"] == ptid]

            # add the current row to the end of the static few-shot examples
            combined = pd.concat([sampled_df, curr_patient_hidden])[["PTID"] + rois + ["AlzheimersDisease"]]

            # shuffle columns if in training
            # if split == 'train':
            #     cols = combined.columns.tolist()
            #     if 'AlzheimersDisease' in cols:
            #         cols.remove('AlzheimersDisease')
            #         shuffled_cols = list(rng.permutation(cols))
            #         shuffled_cols.append('AlzheimersDisease')
            #         combined = combined[shuffled_cols]

            table = combined.head(len(combined)).to_string(index=False)

            # make tabular prompt
            prompt = f"""{instruction_header}
            ### Instruction: {few_shot_instruction}
            ### Input:\n{table}
            ### Response: """
            prompts.append(prompt)

            # create a new dataframe with the prompts and labels
            out_rows.append({
                "ID": ptid,
                "prompt": prompt,
                "AlzheimersDisease": int(curr_df.loc[curr_df["PTID"] == ptid, "AlzheimersDisease"].iloc[0]),
                "split": split,
            })

    return pd.DataFrame(out_rows)

def create_zero_shot_serialized(instruction_header, zero_shot_instruction, df, cols_to_exclude):
    covariates = ["Age", "Education", "Gender", "APOE4"]
    excluded_cols = set(covariates) | set(cols_to_exclude) | set(["AlzheimersDisease"])

    prompts = []
    for _, row in df.iterrows():
        roi_string = "; ".join(
            f"{col}={row[col]:.3f}"
            for col in df.columns
            if col not in excluded_cols
        )

        if pd.isna(row["Gender"]):
            serialized_information = (
                f"A patient has arrived for Alzheimer's Disease (AD) diagnosis."
                f"They have the following imaging regional "
                f"summaries (ROI=value): {roi_string}"
            )
        else:
            if row["Gender"] == 0:
                noun = "man"
                pronoun = "He has"
            elif row["Gender"] == 1:
                noun = "woman"
                pronoun = "She has"
            else:
                noun = "person"
                pronoun = "They have"

            serialized_information = (
                f"A {row['Age']} year old {noun} with {row['APOE4']} copies of APOE4 "
                f"and {row['Education']} years of education arrives for Alzheimer's "
                f"Disease (AD) diagnosis. {pronoun} the following imaging regional "
                f"summaries (ROI=value): {roi_string}"
            )

        prompts.append(f"""{instruction_header}
### Instruction: {zero_shot_instruction}
### Input:\n{serialized_information}
### Response: """)

    new_df = pd.DataFrame({'ID': df['PTID'],
                           'prompt': prompts,
                           'AlzheimersDisease': df['AlzheimersDisease'],
                           'split': df['split']})

    return new_df

def create_few_shot_serialized(instruction_header, few_shot_instruction, df, random_state, k, cols_to_exclude):
    covariates = ["Age", "Education", "Gender", "APOE4"]
    excluded_cols = set(covariates) | set(cols_to_exclude) | set(["AlzheimersDisease"])

    rng = np.random.default_rng(random_state)

    prompts = []
    ids = []
    diag = []
    splits = []

    for split in ['train', 'val', 'test']:
        # sample k examples from the pool for the few-shot examples
        pool = df[df['split'] == f"{split}_pool"]
        pool_ad = pool[pool["AlzheimersDisease"] == 1]
        curr_df = df[df['split'] == split]

        # copy df to hide the labels
        hidden_df = curr_df.copy()
        hidden_df['AlzheimersDisease'] = 'X'
        hidden_df = hidden_df.drop(columns=cols_to_exclude)

        for i in range(len(hidden_df)):
            # guarantee we get at least one AD example in the ICL examples
            sampled_ad = pool_ad.sample(n=1, random_state=int(rng.integers(0, 2 ** 32 - 1)))
            sampled_rest = pool.drop(index=sampled_ad.index).sample(n=k - 1,
                                                                    random_state=int(rng.integers(0, 2 ** 32 - 1)))
            sampled_df = pd.concat([sampled_ad, sampled_rest], ignore_index=True).sample(
                frac=1, random_state=int(rng.integers(0, 2 ** 32 - 1))
            )

            # add the current row to the end of the static few-shot examples
            combined = pd.concat([sampled_df, hidden_df.iloc[[i]]], ignore_index=True)
            combined = combined.drop(columns=cols_to_exclude, errors="ignore")

            serializations = []
            for _, r in combined.iterrows():
                roi_string = "; ".join(
                    f"{col}={r[col]}"
                    for col in combined.columns
                    if col not in excluded_cols
                )

                if r['AlzheimersDisease'] == 1:
                    diagnosis = " and is diagnosed with Alzheimer's disease"
                elif r['AlzheimersDisease'] == 0:
                    diagnosis = " and is not diagnosed with Alzheimer's disease"
                else:
                    diagnosis = ", predict their diagnosis"

                if pd.isna(r["Gender"]):
                    serializations.append(
                        f"A patient has arrived for Alzheimer's Disease (AD) diagnosis{diagnosis}."
                        f"They have the following imaging regional "
                        f"summaries (ROI=value): {roi_string}"
                    )
                else:
                    if r["Gender"] == 0:
                        noun = "man"
                        pronoun = "He has"
                    elif r["Gender"] == 1:
                        noun = "woman"
                        pronoun = "She has"
                    else:
                        noun = "person"
                        pronoun = "They have"

                    serializations.append(
                        f"A {r['Age']} year old {noun} with {r['APOE4']} copies of APOE4 "
                        f"and {r['Education']} years of education arrives for Alzheimer's "
                        f"Disease (AD) diagnosis{diagnosis}. {pronoun} the following imaging regional "
                        f"summaries (ROI=value): {roi_string}"
                    )

            prompts.append(f"""{instruction_header}
### Instruction: {few_shot_instruction}
### Input:
{"\n".join(serializations)}
### Response: """)
            ids.append(curr_df.iloc[i]["PTID"])
            diag.append(curr_df.iloc[i]["AlzheimersDisease"])
            splits.append(split)

    return pd.DataFrame({"ID": ids, "prompt": prompts, "AlzheimersDisease": diag, "split": splits})
