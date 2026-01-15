import time
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    classification_report, average_precision_score, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from src.features.text_url_features import build_text_url_featurizer

DATA_CSV = Path("data/processed/all_emails.csv")
ART = Path("models/artifacts"); ART.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = Path("models/reports"); REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def load_df():
    print(f"Loading dataset from {DATA_CSV} ...")
    return pd.read_csv(DATA_CSV)

def main():
    total_start = time.time()
    df = load_df()
    print("\nLabel counts:\n", df["label"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df, df["label"].values, test_size=0.15, stratify=df["label"], random_state=42
    )

    print("\n[INFO] Starting feature extraction...")
    t0 = time.time()
    ftr = build_text_url_featurizer()
    Xtr = ftr.fit_transform(X_train)
    Xte = ftr.transform(X_test)
    print(f"[TIME] Feature extraction: {time.time() - t0:.2f}s")

    t0 = time.time()
    clf = SGDClassifier(
        loss="log_loss",
        class_weight="balanced",
        max_iter=20,
        tol=1e-3,
        n_jobs=-1,
        random_state=42
    )
    clf.fit(Xtr, y_train)
    print(f"[TIME] Training completed: {time.time() - t0:.2f}s")

    print("\n[INFO] Evaluating model...")
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)

    report = classification_report(y_test, pred, digits=4)
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)

    print(report)

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    cm = confusion_matrix(y_test, pred)

    print("Confusion matrix:\n", cm)
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)
    print(f"ROC-AUC: {roc_auc:.6f}")
    print(f"PR-AUC : {pr_auc:.6f}")

    # save model i featurizer
    joblib.dump(clf, ART / "sgd_lr.joblib")
    joblib.dump(ftr, ART / "text_url_featurizer_sgd.joblib")
    print(f"\n[INFO] Saved artifacts to: {ART}")

    # save izvestaj
    with open(REPORTS_DIR / "sgd_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\nConfusion matrix:\n")
        f.write(str(cm) + "\n")
        f.write(f"\nAccuracy: {acc}\n")
        f.write(f"Precision: {prec}\n")
        f.write(f"Recall: {rec}\n")
        f.write(f"F1 Score: {f1}\n")
        f.write(f"ROC-AUC: {roc_auc:.6f}\n")
        f.write(f"PR-AUC: {pr_auc:.6f}\n")

    print(f"[INFO] Report saved to: {REPORTS_DIR / 'sgd_report.txt'}")
    print(f"\n[TOTAL TIME] {time.time() - total_start:.2f}s")

if __name__ == "__main__":
    main()
