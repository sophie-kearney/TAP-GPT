###
# DATA ABSTRACTION
###
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import sys, random, numpy as np
import pandas as pd

###
# TRADITONAL ML MODELS
###

def run_traditional_ML_models(df, seed, k, p, modality):
    rng = np.random.default_rng(seed)

    # define models
    hyperparameters = pd.read_csv(f"data/hyperparameter_tuning/traditional_ml_results{seed}_{modality}.csv")

    XGB_params = {
        'n_estimators': int(hyperparameters.loc[hyperparameters['model'] == "XGB", 'n_estimators'].values[0]),
        'max_depth': int(hyperparameters.loc[hyperparameters['model'] == "XGB", 'max_depth'].values[0]),
        'learning_rate': float(hyperparameters.loc[hyperparameters['model'] == "XGB", 'learning_rate'].values[0]),
        'subsample': float(hyperparameters.loc[hyperparameters['model'] == "XGB", 'subsample'].values[0]),
        'colsample_bytree': float(hyperparameters.loc[hyperparameters['model'] == "XGB", 'colsample_bytree'].values[0]),
        'gamma': float(hyperparameters.loc[hyperparameters['model'] == "XGB", 'gamma'].values[0]),
        'reg_alpha': float(hyperparameters.loc[hyperparameters['model'] == "XGB", 'reg_alpha'].values[0]),
        'reg_lambda': float(hyperparameters.loc[hyperparameters['model'] == "XGB", 'reg_lambda'].values[0]),
        'random_state': seed,
        'n_jobs': -1
    }

    models = {
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                C=float(hyperparameters.loc[hyperparameters['model'] == "LogisticRegression", 'C'].values[0]),
                penalty=hyperparameters.loc[hyperparameters['model'] == "LogisticRegression", 'penalty'].values[0],
                solver=hyperparameters.loc[hyperparameters['model'] == "LogisticRegression", 'solver'].values[0],
                max_iter=int(
                    hyperparameters.loc[hyperparameters['model'] == "LogisticRegression", 'max_iter'].values[0]),
                random_state=seed
            ))
        ]),
        'RandomForest': RandomForestClassifier(
            n_estimators=int(hyperparameters.loc[hyperparameters['model'] == "RandomForest", 'n_estimators'].values[0]),
            max_depth=int(hyperparameters.loc[hyperparameters['model'] == "RandomForest", 'max_depth'].values[0]),
            min_samples_split=int(
                hyperparameters.loc[hyperparameters['model'] == "RandomForest", 'min_samples_split'].values[0]),
            min_samples_leaf=int(
                hyperparameters.loc[hyperparameters['model'] == "RandomForest", 'min_samples_leaf'].values[0]),
            max_features=hyperparameters.loc[hyperparameters['model'] == "RandomForest", 'max_features'].values[0],
            bootstrap=bool(hyperparameters.loc[hyperparameters['model'] == "RandomForest", 'bootstrap'].values[0]),
            random_state=seed,
            n_jobs=-1
        ),
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(
                C=float(hyperparameters.loc[hyperparameters['model'] == "SVM", 'C'].values[0]),
                kernel=hyperparameters.loc[hyperparameters['model'] == "SVM", 'kernel'].values[0],
                gamma=hyperparameters.loc[hyperparameters['model'] == "SVM", 'gamma'].values[0],
                degree=3,
                probability=True,
                random_state=seed
            ))
        ]),
        'XGBoost': XGBClassifier(**XGB_params)
    }

    # split data
    test_df = df[df['split'] == "test"].fillna(0)
    test_df_pool = df[df['split'] == "test_pool"].fillna(0)
    pool_ad = test_df_pool[test_df_pool["AlzheimersDisease"] == 1]

    # get predictions
    all_predictions = {model: {'y_true': [], 'y_pred': [], 'y_proba': []} for model in models}
    for i in range(len(test_df)):
        # guarantee we get at least one AD example in the ICL examples
        sampled_ad = pool_ad.sample(n=1, random_state=int(rng.integers(0, 2 ** 32 - 1)))
        sampled_rest = test_df_pool.drop(index=sampled_ad.index).sample(n=k - 1, random_state=int(rng.integers(0, 2 ** 32 - 1)))
        sampled_df = pd.concat([sampled_ad, sampled_rest], ignore_index=True).sample(
            frac=1, random_state=int(rng.integers(0, 2 ** 32 - 1))
        )

        y_train = sampled_df['AlzheimersDisease'].values
        x_train = sampled_df.drop(columns=['AlzheimersDisease', 'PTID', 'split'])

        x_test = test_df.drop(columns=['AlzheimersDisease', 'PTID', 'split']).iloc[[i]]
        y_true = test_df['AlzheimersDisease'].iloc[i]

        for model_name, model in models.items():
            if len(np.unique(y_train)) == 1:
                continue

            model.fit(x_train, y_train)
            y_pred_proba = model.predict_proba(x_test)[:, 1][0]
            y_pred = int(y_pred_proba >= 0.5)

            all_predictions[model_name]['y_true'].append(y_true)
            all_predictions[model_name]['y_pred'].append(y_pred)
            all_predictions[model_name]['y_proba'].append(y_pred_proba)

    return all_predictions