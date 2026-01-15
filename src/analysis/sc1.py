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
def plot_decision_boundary_real_model(model, featurizer, df, filename):
    print(">> Generating decision boundary...")

    # priprema podataka
    X_raw = df.drop(columns=["label"])
    y = df["label"].values

    # transformacija u vektore (TF-IDF + URL fitur)
    X_vec = featurizer.transform(X_raw)

    # PCA na 2D da bismo mogli da vizualizujemo
    pca = PCA(n_components=2)
    X_vis = pca.fit_transform(X_vec.toarray())

    # pravi mesh-grid
    x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
    y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]

    # vrati mesh grid nazad u originalni vektorski prostor
    grid_orig = pca.inverse_transform(grid)

    # predikcije modela
    if hasattr(model, "predict_proba"):
        Z = model.predict_proba(grid_orig)[:, 1] >= 0.5
    else:
        Z = model.decision_function(grid_orig) >= 0

    Z = Z.reshape(xx.shape)

    # crtanje
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.35, cmap="bwr")

    # tačke podataka
    plt.scatter(
        X_vis[y == 0, 0], X_vis[y == 0, 1],
        c="blue", edgecolors="k", label="Legitimno"
    )
    plt.scatter(
        X_vis[y == 1, 0], X_vis[y == 1, 1],
        c="red", edgecolors="k", label="Fišing"
    )

    plt.title("Linear Decision Boundary – Phishing Model (PCA)")
    plt.xlabel("PCA komponenta 1")
    plt.ylabel("PCA komponenta 2")
    plt.legend()
    plt.tight_layout()

    # snimi PNG
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
    "decision_boundary_final.png"
)

print(">> Done.")
