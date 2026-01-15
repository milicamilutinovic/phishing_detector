import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    auc, classification_report
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

print(">> Running visualize_results_final.py")

# ==========================================
# Load dataset
# ==========================================
DATA_CSV = "data/processed/all_emails.csv"
print(f">> Loading dataset: {DATA_CSV}")
df = pd.read_csv(DATA_CSV, low_memory=False)

# ==========================================
# Load model + featurizer
# ==========================================
print(">> Loading model artifacts...")
clf = joblib.load("models/artifacts/sgd_lr.joblib")
ftr = joblib.load("models/artifacts/text_url_featurizer_sgd.joblib")

# ==========================================
# Train/test split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    df, df["label"], test_size=0.15, stratify=df["label"], random_state=42
)

# transform
Xte = ftr.transform(X_test)

# scores
scores = clf.predict_proba(Xte)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(Xte)
preds = (scores >= 0.5).astype(int)

# ==========================================
# 1. Matrica konfuzije
# ==========================================
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Legitimno", "Fišing"],
            yticklabels=["Legitimno", "Fišing"])
plt.title("Matrica konfuzije")
plt.xlabel("Predikovana klasa")
plt.ylabel("Stvarna klasa")
plt.tight_layout()
plt.savefig("models/reports/confusion_matrix_final.png")
plt.close()

print("✔ confusion_matrix_final.png")

# ==========================================
# 2. ROC kriva
# ==========================================
fpr, tpr, _ = roc_curve(y_test, scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC kriva")
plt.xlabel("Stopa lažno pozitivnih (FPR)")
plt.ylabel("Stopa istinsko pozitivnih (TPR)")
plt.legend()
plt.tight_layout()
plt.savefig("models/reports/roc_curve_final.png")
plt.close()

print("✔ roc_curve_final.png")

# ==========================================
# 3. Precision–Recall Curve
# ==========================================
prec, rec, _ = precision_recall_curve(y_test, scores)
pr_auc = auc(rec, prec)

plt.figure(figsize=(6, 5))
plt.plot(rec, prec, label=f"AUC = {pr_auc:.4f}")
plt.title("Precision–Recall kriva")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()
plt.savefig("models/reports/precision_recall_curve_final.png")
plt.close()

print("✔ precision_recall_curve_final.png")
