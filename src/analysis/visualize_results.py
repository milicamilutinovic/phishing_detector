import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc


# load dataset
DATA_CSV = "data/processed/all_emails.csv"
df = pd.read_csv(DATA_CSV)


# load model and featurizer
clf = joblib.load("models/artifacts/sgd_lr.joblib")
ftr = joblib.load("models/artifacts/text_url_featurizer_sgd.joblib")

# train/test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df, df["label"], test_size=0.15, stratify=df["label"], random_state=42
)

# transform test set
Xte = ftr.transform(X_test)

# predikcije
if hasattr(clf, "predict_proba"):
    scores = clf.predict_proba(Xte)[:, 1]
else:
    scores = clf.decision_function(Xte)

preds = (scores >= 0.5).astype(int)

# confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimno", "Fišing"],
    yticklabels=["Legitimno", "Fišing"]
)

plt.title("Matrica konfuzije")
plt.ylabel("Stvarna klasa")
plt.xlabel("Predikovana klasa")
plt.tight_layout()

plt.savefig("models/reports/confusion_matrix_new.png")
plt.close()


# roc kriva
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

plt.savefig("models/reports/roc_curve_new.png")
plt.close()

# precision/recall kriva 
prec, rec, _ = precision_recall_curve(y_test, scores)
pr_auc = auc(rec, prec)

plt.figure(figsize=(6, 5))
plt.plot(rec, prec, label=f"AUC = {pr_auc:.4f}")
plt.title("Precision–Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()
plt.savefig("models/reports/precision_recall_curve.png")
plt.close()

# histogram scoreva
plt.figure(figsize=(6, 5))
plt.hist(scores[y_test == 0], bins=40, alpha=0.6, label="Legit")
plt.hist(scores[y_test == 1], bins=40, alpha=0.6, label="Phish")
plt.title("Distribution of Model Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("models/reports/score_distribution.png")
plt.close()



preproc = ftr.named_steps["preproc"]
tfidf = preproc.named_transformers_["tfidf"]

# TF-IDF feature names
tfidf_features = tfidf.get_feature_names_out()

# numeric features from ColumnTransformer
num_features = ['num_urls', 'num_ip_urls', 'avg_url_len', 'max_url_len']

# combine all feature names 
feature_names = np.concatenate([tfidf_features, num_features])

# model coefficients
coefs = clf.coef_[0]

top_n = 20
idx = np.argsort(np.abs(coefs))[-top_n:]

plt.figure(figsize=(10, 8))
plt.barh(feature_names[idx], coefs[idx])
plt.title("Top 20 Most Important Features")
plt.tight_layout()
plt.savefig("models/reports/feature_importance.png")
plt.close()



#calibration plot
from sklearn.calibration import calibration_curve

prob_true, prob_pred = calibration_curve(y_test, scores, n_bins=10)

plt.plot(prob_pred, prob_true, marker="o")
plt.plot([0,1],[0,1],"--")
plt.title("Calibration Plot")
plt.xlabel("Predicted Probability")
plt.ylabel("True Probability")
plt.savefig("models/reports/calibration_plot.png")
plt.close()

#threshold tuning curve
thresholds = np.linspace(0,1,100)
precisions = []
recalls = []

for t in thresholds:
    preds_t = (scores >= t).astype(int)
    cm = confusion_matrix(y_test, preds_t)
    tp = cm[1,1]
    fp = cm[0,1]
    fn = cm[1,0]
    precisions.append(tp/(tp+fp+1e-6))
    recalls.append(tp/(tp+fn+1e-6))

plt.plot(thresholds, precisions, label="Precision")
plt.plot(thresholds, recalls, label="Recall")
plt.legend()
plt.title("Precision/Recall vs Threshold")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.savefig("models/reports/threshold_tuning.png")
plt.close()


#distribution of Email Length
df["body_len"] = df["body"].astype(str).apply(len)

plt.hist(df[df.label==1]["body_len"], alpha=0.6, bins=50, label="Phish")
plt.hist(df[df.label==0]["body_len"], alpha=0.6, bins=50, label="Legit")
plt.legend()
plt.title("Distribution of Email Body Length")
plt.savefig("models/reports/email_length.png")
plt.close()

#URL count distribution
df["url_count"] = df["body"].str.count("http")

plt.hist(df[df.label==1]["url_count"], bins=30, alpha=0.6, label="Phish")
plt.hist(df[df.label==0]["url_count"], bins=30, alpha=0.6, label="Legit")
plt.legend()
plt.title("URL Count Comparison")
plt.savefig("models/reports/url_count_distribution.png")
plt.close()

