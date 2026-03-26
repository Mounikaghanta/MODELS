
# CELL 3 — EVALUATION
#  test_balanced_faults.xlsx
# ============================================================

import numpy as np, pandas as pd, warnings
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
warnings.filterwarnings('ignore')

gt   = pd.read_excel('/content/testmultisensor_faults.xlsx')
pred = pd.read_csv('/content/TST predicted_labels.csv')
gt['Date Time']   = pd.to_datetime(gt['Date Time'],   dayfirst=True)
pred['Date Time'] = pd.to_datetime(pred['Date Time'])

merged = pd.merge(
    pred[['Date Time','predicted_label','p_faulty']],
    gt[['Date Time','binary_label','fault_label']],
    on='Date Time', how='inner'
)
y_true = merged['binary_label'].values
y_pred = merged['predicted_label'].values
y_prob = merged['p_faulty'].values
ftype  = merged['fault_label'].values

# ── Metrics ──────────────────────────────────────────────────
acc     = accuracy_score(y_true, y_pred)
f1      = f1_score(y_true, y_pred, zero_division=0)
prec    = precision_score(y_true, y_pred, zero_division=0)
rec     = recall_score(y_true, y_pred, zero_division=0)
roc     = roc_auc_score(y_true, y_prob)
ap      = average_precision_score(y_true, y_prob)
tn,fp,fn,tp = confusion_matrix(y_true, y_pred).ravel()


print('   TST BINARY FAULT DETECTION — RESULTS')
print('═'*55)
print(f'  Accuracy          : {acc:.4f}  ({acc:.2%})')
print(f'  F1 (faulty=pos)   : {f1:.4f}')
print(f'  Precision         : {prec:.4f}')
print(f'  Recall            : {rec:.4f}  ← % faults caught')
print(f'  TN={tn} FP={fp} FN={fn} TP={tp}')
