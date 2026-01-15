import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_blobs

# Generate synthetic linearly separable dataset
X, y = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.2)

# Fit linear model
clf = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3)
clf.fit(X, y)

# Create mesh grid for background coloring
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))

# Predict on grid
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = clf.predict(grid_points).reshape(xx.shape)

# Plot background
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="bwr")

# Plot points
plt.scatter(X[y==0, 0], X[y==0, 1], c="blue", edgecolors="k", label="Class 1")
plt.scatter(X[y==1, 0], X[y==1, 1], c="red", edgecolors="k", label="Class 2")

# Plot decision boundary line
w = clf.coef_[0]
b = clf.intercept_[0]
x_vals = np.linspace(x_min, x_max, 200)
y_vals = -(w[0] * x_vals + b) / w[1]
plt.plot(x_vals, y_vals, "k--", linewidth=2)

plt.title("Linear Decision Boundary (SGDClassifier Example)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.tight_layout()

plt.savefig("models/reports/linear_decision_boundary_example.png", dpi=300)
plt.show()
