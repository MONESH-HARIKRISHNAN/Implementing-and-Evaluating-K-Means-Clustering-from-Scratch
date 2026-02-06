# K-Means Clustering From Scratch (NumPy Only) + Silhouette Evaluation
# Imports + Reproducibility
import numpy as np
import matplotlib.pyplot as plt

# Evaluation metric only (allowed)
from sklearn.metrics import silhouette_score

np.random.seed(7)

# Create a Synthetic 2D Dataset (4 clear clusters)
# ---- Synthetic dataset ----
n_per_cluster = 120

centers = np.array([
    [-2.5, -2.0],
    [ 3.5,  3.0],
    [-2.0,  3.5],
    [ 4.0, -2.5]
])

spread = 0.75  # controls how tight each cluster is

clouds = []
for c in centers:
    points = np.random.randn(n_per_cluster, 2) * spread + c
    clouds.append(points)

data = np.vstack(clouds)

print("Dataset shape:", data.shape)

plt.figure(figsize=(6, 5))
plt.scatter(data[:, 0], data[:, 1], s=18)
plt.title("Synthetic 2D Dataset (Unlabeled)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(alpha=0.25)
plt.show()

# K-Means Implementation From Scratch (NumPy)
class NumpyKMeans:
    def __init__(self, num_clusters=3, max_steps=120, tol=1e-4):
        self.num_clusters = num_clusters
        self.max_steps = max_steps
        self.tol = tol

        self.centers_ = None
        self.cluster_id_ = None

    def _init_centers(self, data):
        # pick K unique points from the dataset as starting centers
        idx = np.random.choice(len(data), self.num_clusters, replace=False)
        return data[idx].copy()

    def _pairwise_distances(self, data, centers):
        # returns a matrix of shape (n_samples, K)
        # each entry is Euclidean distance between point and center
        return np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)

    def _assign(self, data, centers):
        dist = self._pairwise_distances(data, centers)
        return np.argmin(dist, axis=1)

    def _recompute_centers(self, data, cluster_id):
        new_centers = np.zeros((self.num_clusters, data.shape[1]), dtype=float)

        for k in range(self.num_clusters):
            members = data[cluster_id == k]

            # If a cluster becomes empty, reinitialize its center randomly
            if len(members) == 0:
                new_centers[k] = data[np.random.randint(0, len(data))]
            else:
                new_centers[k] = members.mean(axis=0)

        return new_centers

    def fit(self, data):
        centers = self._init_centers(data)

        for step in range(self.max_steps):
            cluster_id = self._assign(data, centers)
            new_centers = self._recompute_centers(data, cluster_id)

            # Convergence: if centers barely move, stop
            movement = np.linalg.norm(new_centers - centers)
            centers = new_centers

            if movement < self.tol:
                break

        self.centers_ = centers
        self.cluster_id_ = cluster_id
        self.steps_used_ = step + 1

        return self

# Run K-Means for K = 2 to 6 and Compute Silhouette Scores
scores = {}
trained_models = {}

for k in range(2, 7):
    model = NumpyKMeans(num_clusters=k, max_steps=200, tol=1e-4).fit(data)
    labels = model.cluster_id_

    s = silhouette_score(data, labels)  # evaluation only
    scores[k] = s
    trained_models[k] = model

    print(f"K={k} | steps={model.steps_used_:3d} | silhouette={s:.4f}")

# Plot Silhouette Score vs K
ks = np.array(list(scores.keys()))
vals = np.array(list(scores.values()))

best_k = int(ks[np.argmax(vals)])
print("Best K based on silhouette:", best_k)

plt.figure(figsize=(6, 4))
plt.plot(ks, vals, marker="o")
plt.title("Silhouette Score vs K")
plt.xlabel("K (number of clusters)")
plt.ylabel("Silhouette score")
plt.grid(alpha=0.3)
plt.show()

# Visualize Final Clustering for the Best K
best_model = trained_models[best_k]

plt.figure(figsize=(6, 5))
plt.scatter(data[:, 0], data[:, 1], c=best_model.cluster_id_, s=18, cmap="viridis")
plt.scatter(best_model.centers_[:, 0], best_model.centers_[:, 1], s=220, marker="X", c="red")
plt.title(f"K-Means Clustering Result (K={best_k})")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(alpha=0.25)
plt.show()
