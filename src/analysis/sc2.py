import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# ===========================
# Load dataset
# ===========================
DATA_CSV = "data/processed/all_emails.csv"
df = pd.read_csv(DATA_CSV)
print(f">> Loaded dataset: {DATA_CSV}")


# ===========================
# Load trained model + featurizer
# ===========================
clf = joblib.load("models/artifacts/sgd_lr.joblib")
ftr = joblib.load("models/artifacts/text_url_featurizer_sgd.joblib")
print(">> Loaded model + featurizer")


# ===========================
# Function: Decision Boundary
# ===========================
from sklearn.decomposition import PCA

def plot_decision_boundary_real_model(model, featurizer, df, filename):
    print(">> Generating FAST decision boundary...")

    # Koristi samo 3000 nasumičnih emailova za PCA (dovoljno za vizualizaciju)
    df_sample = df.sample(n=3000, random_state=42)

    X_raw = df_sample.drop(columns=["label"])
    y = df_sample["label"].values

    # Transformacija
    X_vec = featurizer.transform(X_raw)

    # PCA na sparse matrici (radi brzo)
    pca = PCA(n_components=2)
    X_vis = pca.fit_transform(X_vec.toarray())

    # Mesh grid (manji)
    x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
    y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),   # 200 umesto 300
        np.linspace(y_min, y_max, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    # Rekonstrukcija PCA → znatno manja dimenzionalnost
    grid_orig = pca.inverse_transform(grid)

    # Predikcija
    if hasattr(model, "predict_proba"):
        Z = model.predict_proba(grid_orig)[:, 1] >= 0.5
    else:
        Z = model.decision_function(grid_orig) >= 0

    Z = Z.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.35, cmap="bwr")

    plt.scatter(
        X_vis[y == 0, 0], X_vis[y == 0, 1],
        c="blue", edgecolors="k", label="Legitimno"
    )
    plt.scatter(
        X_vis[y == 1, 0], X_vis[y == 1, 1],
        c="red", edgecolors="k", label="Fišing"
    )

    plt.title("Linear Decision Boundary – Phishing Model (FAST PCA)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.tight_layout()

    out_path = f"models/reports/{filename}"
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✔ Saved: {out_path}")

# ===========================
# Call plotting function
# ===========================
plot_decision_boundary_real_model(
    clf,
    ftr,
    df,
    "decision_boundary_final1.png"
)

print(">> Done.")
