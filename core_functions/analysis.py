###
# DATA ABSTRACTION
###

import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, precision_score, recall_score, confusion_matrix,
    balanced_accuracy_score
)
from datetime import datetime
import os, sys

###
# ANALYSIS
###

def get_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # print(f"Confusion matrix:\nTN={tn}, FP={fp}, FN={fn}, TP={tp}\n")

    return {"F1": round(f1_score(y_true, y_pred), 4),
            "Accuracy": round(accuracy_score(y_true, y_pred), 4),
            "BalancedAccuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
            "Precision": round(precision_score(y_true, y_pred), 4),
            "Recall": round(recall_score(y_true, y_pred), 4),
            "Specificity": round(tn / (tn + fp), 4)}