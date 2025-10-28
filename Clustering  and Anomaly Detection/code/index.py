# index.py — Clustering & Anomaly Detection on Iris (Python)
# Fixes: robust cluster→class mapping (classes x clusters), no square-matrix assumptions.

import os
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix
from sklearn.ensemble import IsolationForest

import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go

from scipy.optimize import linear_sum_assignment


# -------------------------------
# Helpers
# -------------------------------

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def elbow_wss(X, kmax=10, random_state=123):
    wss = []
    for k in range(1, kmax + 1):
        km = KMeans(n_clusters=k, n_init=25, random_state=random_state).fit(X)
        wss.append(km.inertia_)
    return np.arange(1, kmax + 1), np.array(wss)


def silhouette_over_k(X, kmin=2, kmax=10, random_state=123):
    sil = []
    ks = range(kmin, kmax + 1)
    for k in ks:
        km = KMeans(n_clusters=k, n_init=25, random_state=random_state).fit(X)
        sil.append(silhouette_score(X, km.labels_))
    return np.array(list(ks)), np.array(sil)


def predict_kmeans_labels(X, centers):
    # Assign by nearest center (Euclidean)
    dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(dists, axis=1)


def knn_avg_distance(X, k=5):
    # Average distance to k nearest neighbors (excluding self)
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(X)
    dists, _ = nn.kneighbors(X)  # includes self at [:,0]==0
    return dists[:, 1:].mean(axis=1)


def pca_fit_transform(X_train, X_test, n_components=2):
    pca = PCA(n_components=n_components, svd_solver='full', random_state=123)
    return pca, pca.fit_transform(X_train), pca.transform(X_test)


def best_cluster_label_mapping(y_true, y_pred):
    """
    Map cluster ids (in y_pred) to class labels (in y_true) to maximize accuracy.
    Builds a contingency matrix of shape [n_classes x n_clusters], then runs Hungarian.
    Returns:
      mapping: dict {cluster_id -> class_label}
      acc: accuracy after mapping (on y_true,y_pred)
      cm: contingency matrix (classes x clusters)
    """
    classes = np.unique(y_true)
    clusters = np.unique(y_pred)

    # contingency: rows=classes, cols=clusters
    cm = np.zeros((len(classes), len(clusters)), dtype=int)
    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            cm[i, j] = np.sum((y_true == c) & (y_pred == k))

    # Hungarian on -cm (maximize)
    row_ind, col_ind = linear_sum_assignment(-cm)

    # Build mapping cluster -> class
    mapping = {clusters[j]: classes[i] for i, j in zip(row_ind, col_ind)}

    # Compute accuracy using mapping
    mapped = np.vectorize(lambda c: mapping.get(c, classes[0]))(y_pred)
    acc = (mapped == y_true).mean()
    return mapping, acc, cm


def plot_confusion_matrix(cm, xlabels, ylabels, title, outpath):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(cm, interpolation='nearest', aspect='auto')
    ax.set_title(title)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticklabels(ylabels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# -------------------------------
# Load & preprocess
# -------------------------------

iris = load_iris(as_frame=True)
df = iris.frame.copy()
df.rename(columns={
    'sepal length (cm)': 'Sepal.Length',
    'sepal width (cm)': 'Sepal.Width',
    'petal length (cm)': 'Petal.Length',
    'petal width (cm)': 'Petal.Width',
    'target': 'Species'
}, inplace=True)
df['Species'] = df['Species'].map(dict(zip(range(3), iris.target_names)))
species_order = np.array(iris.target_names)

X = df[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']].to_numpy(dtype=float)
y = df['Species'].to_numpy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=123, stratify=y
)

pca, pca_train, pca_test = pca_fit_transform(X_train, X_test, n_components=2)

# -------------------------------
# K-Means: elbow + silhouette
# -------------------------------

ks_wss, wss = elbow_wss(X_train, kmax=10, random_state=123)
ks_sil, sil = silhouette_over_k(X_train, kmin=2, kmax=10, random_state=123)
optimal_k = int(ks_sil[np.argmax(sil)])

kmeans = KMeans(n_clusters=optimal_k, n_init=25, random_state=123).fit(X_train)
train_kmeans_labels = kmeans.labels_
test_kmeans_labels = predict_kmeans_labels(X_test, kmeans.cluster_centers_)

sil_test = silhouette_score(X_test, test_kmeans_labels)
kmeans_ari = adjusted_rand_score(y_train, train_kmeans_labels)

# Fixed mapping (classes x clusters contingency)
mapping, kmeans_acc, kmeans_contingency = best_cluster_label_mapping(y_train, train_kmeans_labels)
mapped_train_preds = np.vectorize(lambda c: mapping[c])(train_kmeans_labels)
kmeans_cm_full = confusion_matrix(y_train, mapped_train_preds, labels=species_order)

# -------------------------------
# DBSCAN (eps via kNN heuristic)
# -------------------------------

n_feat = X_train.shape[1]
min_pts = n_feat + 1

nn = NearestNeighbors(n_neighbors=min_pts, metric='euclidean').fit(X_train)
knn_dists, _ = nn.kneighbors(X_train)
knn_dist_sorted = np.sort(knn_dists[:, -1])
eps_value = float(np.quantile(knn_dist_sorted, 0.95))

dbscan_model = DBSCAN(eps=eps_value, min_samples=min_pts, metric='euclidean').fit(X_train)
train_db_labels = dbscan_model.labels_  # noise = -1

mask_non_noise = train_db_labels != -1
if mask_non_noise.sum() > 1 and len(np.unique(train_db_labels[mask_non_noise])) > 1:
    dbscan_sil = silhouette_score(X_train[mask_non_noise], train_db_labels[mask_non_noise])
else:
    dbscan_sil = np.nan

y_train_non_noise = y_train[mask_non_noise]
db_labels_non_noise = train_db_labels[mask_non_noise]
dbscan_ari = adjusted_rand_score(y_train_non_noise, db_labels_non_noise) if mask_non_noise.any() else np.nan
dbscan_noise_count = int((train_db_labels == -1).sum())

# For a mapped confusion matrix (non-noise only)
if mask_non_noise.any():
    db_mapping, db_acc, _ = best_cluster_label_mapping(y_train_non_noise, db_labels_non_noise)
    mapped_db = np.vectorize(lambda c: db_mapping.get(c, species_order[0]))(db_labels_non_noise)
    dbscan_cm_full = confusion_matrix(y_train_non_noise, mapped_db, labels=species_order)
else:
    dbscan_cm_full = None

# -------------------------------
# Outlier detectors (95th pct)
# -------------------------------

# KNN distance outliers
k_value = 5
train_knn_dist = knn_avg_distance(X_train, k=k_value)
knn_threshold = np.quantile(train_knn_dist, 0.95)
train_knn_outlier = np.where(train_knn_dist > knn_threshold, "Outlier", "Normal")

test_knn_dist = knn_avg_distance(X_test, k=k_value)
test_knn_outlier = np.where(test_knn_dist > knn_threshold, "Outlier", "Normal")

# Isolation Forest
iso = IsolationForest(n_estimators=100, max_samples='auto', contamination='auto',
                      random_state=123, n_jobs=1)
iso.fit(X_train)
train_iso_scores = -iso.score_samples(X_train)
iso_threshold = np.quantile(train_iso_scores, 0.95)
train_iso_flag = np.where(train_iso_scores > iso_threshold, "Anomaly", "Normal")

test_iso_scores = -iso.score_samples(X_test)
test_iso_flag = np.where(test_iso_scores > iso_threshold, "Anomaly", "Normal")

# LOF (novelty=True allows scoring test)
lof = LocalOutlierFactor(n_neighbors=6, novelty=True)
lof.fit(X_train)
train_lof_scores = -lof.score_samples(X_train)
lof_threshold = np.quantile(train_lof_scores, 0.95)
train_lof_flag = np.where(train_lof_scores > lof_threshold, "Anomaly", "Normal")

test_lof_scores = -lof.score_samples(X_test)
test_lof_flag = np.where(test_lof_scores > lof_threshold, "Anomaly", "Normal")

# -------------------------------
# Metrics & overlaps
# -------------------------------

kmeans_avg_sil = silhouette_score(X_train, train_kmeans_labels)
dbscan_avg_sil = dbscan_sil
knn_outliers_count = int((train_knn_outlier == "Outlier").sum())
iso_outliers_count = int((train_iso_flag == "Anomaly").sum())
lof_outliers_count = int((train_lof_flag == "Anomaly").sum())
dbscan_noise = dbscan_noise_count

idx_knn = set(np.where(train_knn_outlier == "Outlier")[0])
idx_iso = set(np.where(train_iso_flag == "Anomaly")[0])
idx_lof = set(np.where(train_lof_flag == "Anomaly")[0])
idx_db_noise = set(np.where(train_db_labels == -1)[0])

overlap_knn_iso = len(idx_knn & idx_iso)
overlap_knn_lof = len(idx_knn & idx_lof)
overlap_knn_db = len(idx_knn & idx_db_noise)
overlap_iso_lof = len(idx_iso & idx_lof)
overlap_iso_db = len(idx_iso & idx_db_noise)
overlap_lof_db = len(idx_lof & idx_db_noise)
overlap_all = len(idx_knn & idx_iso & idx_lof & idx_db_noise)

# Per-species anomaly percentages (train)
def anomaly_pct_per_species(method_flags, positive_value):
    out = {}
    for sp in species_order:
        mask = (y_train == sp)
        total = mask.sum()
        out[sp] = 100.0 * (method_flags[mask] == positive_value).sum() / total if total else 0.0
    return out

pct_knn = anomaly_pct_per_species(train_knn_outlier, "Outlier")
pct_iso = anomaly_pct_per_species(train_iso_flag, "Anomaly")
pct_lof = anomaly_pct_per_species(train_lof_flag, "Anomaly")

# -------------------------------
# Output directory & plots
# -------------------------------

out_dir = ensure_dir("Clustering_Anomaly_Detection_Plots")

# Elbow
plt.figure(figsize=(7, 5))
plt.plot(ks_wss, wss, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Within-Cluster Sum of Squares")
plt.title("Elbow Method for Optimal k")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "elbow_plot.png"), dpi=150)
plt.close()

# Silhouette vs k
plt.figure(figsize=(7, 5))
plt.plot(ks_sil, sil, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Average Silhouette Score")
plt.title("Silhouette Method for Optimal k")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "silhouette_plot.png"), dpi=150)
plt.close()

# KMeans PCA
plt.figure(figsize=(7, 5))
for lab in np.unique(train_kmeans_labels):
    m = (train_kmeans_labels == lab)
    plt.scatter(pca_train[m, 0], pca_train[m, 1], alpha=0.7, label=f"Cluster {lab+1}")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("K-means Clustering (PCA)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "kmeans_plot.png"), dpi=150)
plt.close()

# DBSCAN PCA
plt.figure(figsize=(7, 5))
for lab in np.unique(train_db_labels):
    m = (train_db_labels == lab)
    name = "Noise" if lab == -1 else f"Cluster {lab}"
    plt.scatter(pca_train[m, 0], pca_train[m, 1], alpha=0.7, label=name)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("DBSCAN (PCA)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "dbscan_plot.png"), dpi=150)
plt.close()

# KNN outliers PCA
plt.figure(figsize=(7, 5))
mask_norm = (train_knn_outlier == "Normal")
mask_out = (train_knn_outlier == "Outlier")
plt.scatter(pca_train[mask_norm, 0], pca_train[mask_norm, 1], alpha=0.7, label="Normal")
plt.scatter(pca_train[mask_out, 0], pca_train[mask_out, 1], alpha=0.7, label="Outlier")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("KNN Outlier Detection (PCA)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "knn_plot.png"), dpi=150)
plt.close()

# Isolation Forest PCA
plt.figure(figsize=(7, 5))
mask_norm = (train_iso_flag == "Normal")
mask_anom = (train_iso_flag == "Anomaly")
plt.scatter(pca_train[mask_norm, 0], pca_train[mask_norm, 1], alpha=0.7, label="Normal")
plt.scatter(pca_train[mask_anom, 0], pca_train[mask_anom, 1], alpha=0.7, label="Anomaly")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Isolation Forest (PCA)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "iso_plot.png"), dpi=150)
plt.close()

# LOF PCA
plt.figure(figsize=(7, 5))
mask_norm = (train_lof_flag == "Normal")
mask_anom = (train_lof_flag == "Anomaly")
plt.scatter(pca_train[mask_norm, 0], pca_train[mask_norm, 1], alpha=0.7, label="Normal")
plt.scatter(pca_train[mask_anom, 0], pca_train[mask_anom, 1], alpha=0.7, label="Anomaly")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("LOF (PCA)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "lof_plot.png"), dpi=150)
plt.close()

# Confusion matrices (mapped)
plot_confusion_matrix(kmeans_cm_full, species_order, species_order,
                      "K-Means Confusion Matrix (mapped, TRAIN)",
                      os.path.join(out_dir, "confusion_matrix_kmeans.png"))

if dbscan_cm_full is not None:
    plot_confusion_matrix(dbscan_cm_full, species_order, species_order,
                          "DBSCAN Confusion Matrix (mapped, TRAIN, non-noise)",
                          os.path.join(out_dir, "confusion_matrix_dbscan.png"))

# -------------------------------
# Interactive Plotly toggle (TRAIN)
# -------------------------------

fig = go.Figure()

# Actual species
for sp in species_order:
    m = (y_train == sp)
    fig.add_trace(go.Scatter(x=pca_train[m, 0], y=pca_train[m, 1], mode='markers',
                             name=f"Actual: {sp}", marker=dict(size=8, opacity=0.6), visible=True))

# KMeans
for cl in np.unique(train_kmeans_labels):
    m = (train_kmeans_labels == cl)
    fig.add_trace(go.Scatter(x=pca_train[m, 0], y=pca_train[m, 1], mode='markers',
                             name=f"K-means: C{cl+1}", marker=dict(size=8, opacity=0.6), visible=False))

# DBSCAN
for cl in np.unique(train_db_labels):
    m = (train_db_labels == cl)
    nm = "Noise" if cl == -1 else f"DBSCAN: C{cl}"
    fig.add_trace(go.Scatter(x=pca_train[m, 0], y=pca_train[m, 1], mode='markers',
                             name=nm, marker=dict(size=8, opacity=0.6), visible=False))

# KNN
for lab in ["Normal", "Outlier"]:
    m = (train_knn_outlier == lab)
    fig.add_trace(go.Scatter(x=pca_train[m, 0], y=pca_train[m, 1], mode='markers',
                             name=f"KNN: {lab}", marker=dict(size=8, opacity=0.6), visible=False))

# ISO
for lab in ["Normal", "Anomaly"]:
    m = (train_iso_flag == lab)
    fig.add_trace(go.Scatter(x=pca_train[m, 0], y=pca_train[m, 1], mode='markers',
                             name=f"ISO: {lab}", marker=dict(size=8, opacity=0.6), visible=False))

# LOF
for lab in ["Normal", "Anomaly"]:
    m = (train_lof_flag == lab)
    fig.add_trace(go.Scatter(x=pca_train[m, 0], y=pca_train[m, 1], mode='markers',
                             name=f"LOF: {lab}", marker=dict(size=8, opacity=0.6), visible=False))

n_species = len(species_order)
n_kmeans = len(np.unique(train_kmeans_labels))
n_dbscan = len(np.unique(train_db_labels))
n_knn = 2
n_iso = 2
n_lof = 2

vis_actual = [True]*n_species + [False]*(n_kmeans + n_dbscan + n_knn + n_iso + n_lof)
vis_kmeans = [False]*n_species + [True]*n_kmeans + [False]*(n_dbscan + n_knn + n_iso + n_lof)
vis_dbscan = [False]*(n_species + n_kmeans) + [True]*n_dbscan + [False]*(n_knn + n_iso + n_lof)
vis_knn = [False]*(n_species + n_kmeans + n_dbscan) + [True]*n_knn + [False]*(n_iso + n_lof)
vis_iso = [False]*(n_species + n_kmeans + n_dbscan + n_knn) + [True]*n_iso + [False]*n_lof
vis_lof = [False]*(n_species + n_kmeans + n_dbscan + n_knn + n_iso) + [True]*n_lof

fig.update_layout(
    title="Comparison of Clustering and Anomaly Detection Methods (TRAIN, PCA)",
    xaxis_title="PC1", yaxis_title="PC2",
    updatemenus=[dict(
        type="buttons", direction="right", x=0.1, y=1.15,
        buttons=[
            dict(label="Actual Species", method="update", args=[{"visible": vis_actual}]),
            dict(label="K-means", method="update", args=[{"visible": vis_kmeans}]),
            dict(label="DBSCAN", method="update", args=[{"visible": vis_dbscan}]),
            dict(label="KNN", method="update", args=[{"visible": vis_knn}]),
            dict(label="Isolation Forest", method="update", args=[{"visible": vis_iso}]),
            dict(label="LOF", method="update", args=[{"visible": vis_lof}]),
        ]
    )]
)

pio.write_html(fig, file=os.path.join(out_dir, "interactive_comparison.html"), auto_open=False)

# -------------------------------
# Console summary
# -------------------------------

print("Optimal k based on silhouette score:", optimal_k)
print("Silhouette score on test data:", sil_test)
print("\nK-means clustering accuracy (TRAIN, mapped):", kmeans_acc)
print("K-means Adjusted Rand Index (TRAIN):", kmeans_ari)

print("\nDBSCAN average silhouette score (TRAIN, non-noise):", dbscan_avg_sil)
print("DBSCAN Adjusted Rand Index (TRAIN, excluding noise):", dbscan_ari)
print("Number of DBSCAN noise points (TRAIN):", dbscan_noise)

print("\nEvaluation Metrics:")
print("-------------------")
print("K-means average silhouette score (TRAIN):", kmeans_avg_sil)
print("DBSCAN average silhouette score (TRAIN):", dbscan_avg_sil)
print("Number of KNN outliers (TRAIN):", knn_outliers_count)
print("Number of Isolation Forest anomalies (TRAIN):", iso_outliers_count)
print("Number of LOF anomalies (TRAIN):", lof_outliers_count)
print("Number of DBSCAN noise points (TRAIN):", dbscan_noise)

print("\nOverlap between anomaly detection methods (TRAIN):")
print("----------------------------------------")
print("KNN and Isolation Forest:", overlap_knn_iso, "points")
print("KNN and LOF:", overlap_knn_lof, "points")
print("KNN and DBSCAN noise:", overlap_knn_db, "points")
print("Isolation Forest and LOF:", overlap_iso_lof, "points")
print("Isolation Forest and DBSCAN noise:", overlap_iso_db, "points")
print("LOF and DBSCAN noise:", overlap_lof_db, "points")
print("All four methods:", overlap_all, "points")

print("\nPercentage of each species detected as anomalies (TRAIN):")
print("KNN:", {k: f"{v:.2f}%" for k, v in pct_knn.items()})
print("ISO:", {k: f"{v:.2f}%" for k, v in pct_iso.items()})
print("LOF:", {k: f"{v:.2f}%" for k, v in pct_lof.items()})

print(f"\nSaved figures and interactive plot to: {out_dir}")
# index.py — Clustering & Anomaly Detection on Iris (Python)
# Fixes: robust cluster→class mapping (classes x clusters), no square-matrix assumptions.

import os
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.metrics import silhouette_score, adjusted_rand_score, confusion_matrix
from sklearn.ensemble import IsolationForest

import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go

from scipy.optimize import linear_sum_assignment


# -------------------------------
# Helpers
# -------------------------------

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def elbow_wss(X, kmax=10, random_state=123):
    wss = []
    for k in range(1, kmax + 1):
        km = KMeans(n_clusters=k, n_init=25, random_state=random_state).fit(X)
        wss.append(km.inertia_)
    return np.arange(1, kmax + 1), np.array(wss)


def silhouette_over_k(X, kmin=2, kmax=10, random_state=123):
    sil = []
    ks = range(kmin, kmax + 1)
    for k in ks:
        km = KMeans(n_clusters=k, n_init=25, random_state=random_state).fit(X)
        sil.append(silhouette_score(X, km.labels_))
    return np.array(list(ks)), np.array(sil)


def predict_kmeans_labels(X, centers):
    # Assign by nearest center (Euclidean)
    dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(dists, axis=1)


def knn_avg_distance(X, k=5):
    # Average distance to k nearest neighbors (excluding self)
    nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(X)
    dists, _ = nn.kneighbors(X)  # includes self at [:,0]==0
    return dists[:, 1:].mean(axis=1)


def pca_fit_transform(X_train, X_test, n_components=2):
    pca = PCA(n_components=n_components, svd_solver='full', random_state=123)
    return pca, pca.fit_transform(X_train), pca.transform(X_test)


def best_cluster_label_mapping(y_true, y_pred):
    """
    Map cluster ids (in y_pred) to class labels (in y_true) to maximize accuracy.
    Builds a contingency matrix of shape [n_classes x n_clusters], then runs Hungarian.
    Returns:
      mapping: dict {cluster_id -> class_label}
      acc: accuracy after mapping (on y_true,y_pred)
      cm: contingency matrix (classes x clusters)
    """
    classes = np.unique(y_true)
    clusters = np.unique(y_pred)

    # contingency: rows=classes, cols=clusters
    cm = np.zeros((len(classes), len(clusters)), dtype=int)
    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            cm[i, j] = np.sum((y_true == c) & (y_pred == k))

    # Hungarian on -cm (maximize)
    row_ind, col_ind = linear_sum_assignment(-cm)

    # Build mapping cluster -> class
    mapping = {clusters[j]: classes[i] for i, j in zip(row_ind, col_ind)}

    # Compute accuracy using mapping
    mapped = np.vectorize(lambda c: mapping.get(c, classes[0]))(y_pred)
    acc = (mapped == y_true).mean()
    return mapping, acc, cm


def plot_confusion_matrix(cm, xlabels, ylabels, title, outpath):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(cm, interpolation='nearest', aspect='auto')
    ax.set_title(title)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticklabels(ylabels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# -------------------------------
# Load & preprocess
# -------------------------------

iris = load_iris(as_frame=True)
df = iris.frame.copy()
df.rename(columns={
    'sepal length (cm)': 'Sepal.Length',
    'sepal width (cm)': 'Sepal.Width',
    'petal length (cm)': 'Petal.Length',
    'petal width (cm)': 'Petal.Width',
    'target': 'Species'
}, inplace=True)
df['Species'] = df['Species'].map(dict(zip(range(3), iris.target_names)))
species_order = np.array(iris.target_names)

X = df[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']].to_numpy(dtype=float)
y = df['Species'].to_numpy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=123, stratify=y
)

pca, pca_train, pca_test = pca_fit_transform(X_train, X_test, n_components=2)

# -------------------------------
# K-Means: elbow + silhouette
# -------------------------------

ks_wss, wss = elbow_wss(X_train, kmax=10, random_state=123)
ks_sil, sil = silhouette_over_k(X_train, kmin=2, kmax=10, random_state=123)
optimal_k = int(ks_sil[np.argmax(sil)])

kmeans = KMeans(n_clusters=optimal_k, n_init=25, random_state=123).fit(X_train)
train_kmeans_labels = kmeans.labels_
test_kmeans_labels = predict_kmeans_labels(X_test, kmeans.cluster_centers_)

sil_test = silhouette_score(X_test, test_kmeans_labels)
kmeans_ari = adjusted_rand_score(y_train, train_kmeans_labels)

# Fixed mapping (classes x clusters contingency)
mapping, kmeans_acc, kmeans_contingency = best_cluster_label_mapping(y_train, train_kmeans_labels)
mapped_train_preds = np.vectorize(lambda c: mapping[c])(train_kmeans_labels)
kmeans_cm_full = confusion_matrix(y_train, mapped_train_preds, labels=species_order)

# -------------------------------
# DBSCAN (eps via kNN heuristic)
# -------------------------------

n_feat = X_train.shape[1]
min_pts = n_feat + 1

nn = NearestNeighbors(n_neighbors=min_pts, metric='euclidean').fit(X_train)
knn_dists, _ = nn.kneighbors(X_train)
knn_dist_sorted = np.sort(knn_dists[:, -1])
eps_value = float(np.quantile(knn_dist_sorted, 0.95))

dbscan_model = DBSCAN(eps=eps_value, min_samples=min_pts, metric='euclidean').fit(X_train)
train_db_labels = dbscan_model.labels_  # noise = -1

mask_non_noise = train_db_labels != -1
if mask_non_noise.sum() > 1 and len(np.unique(train_db_labels[mask_non_noise])) > 1:
    dbscan_sil = silhouette_score(X_train[mask_non_noise], train_db_labels[mask_non_noise])
else:
    dbscan_sil = np.nan

y_train_non_noise = y_train[mask_non_noise]
db_labels_non_noise = train_db_labels[mask_non_noise]
dbscan_ari = adjusted_rand_score(y_train_non_noise, db_labels_non_noise) if mask_non_noise.any() else np.nan
dbscan_noise_count = int((train_db_labels == -1).sum())

# For a mapped confusion matrix (non-noise only)
if mask_non_noise.any():
    db_mapping, db_acc, _ = best_cluster_label_mapping(y_train_non_noise, db_labels_non_noise)
    mapped_db = np.vectorize(lambda c: db_mapping.get(c, species_order[0]))(db_labels_non_noise)
    dbscan_cm_full = confusion_matrix(y_train_non_noise, mapped_db, labels=species_order)
else:
    dbscan_cm_full = None

# -------------------------------
# Outlier detectors (95th pct)
# -------------------------------

# KNN distance outliers
k_value = 5
train_knn_dist = knn_avg_distance(X_train, k=k_value)
knn_threshold = np.quantile(train_knn_dist, 0.95)
train_knn_outlier = np.where(train_knn_dist > knn_threshold, "Outlier", "Normal")

test_knn_dist = knn_avg_distance(X_test, k=k_value)
test_knn_outlier = np.where(test_knn_dist > knn_threshold, "Outlier", "Normal")

# Isolation Forest
iso = IsolationForest(n_estimators=100, max_samples='auto', contamination='auto',
                      random_state=123, n_jobs=1)
iso.fit(X_train)
train_iso_scores = -iso.score_samples(X_train)
iso_threshold = np.quantile(train_iso_scores, 0.95)
train_iso_flag = np.where(train_iso_scores > iso_threshold, "Anomaly", "Normal")

test_iso_scores = -iso.score_samples(X_test)
test_iso_flag = np.where(test_iso_scores > iso_threshold, "Anomaly", "Normal")

# LOF (novelty=True allows scoring test)
lof = LocalOutlierFactor(n_neighbors=6, novelty=True)
lof.fit(X_train)
train_lof_scores = -lof.score_samples(X_train)
lof_threshold = np.quantile(train_lof_scores, 0.95)
train_lof_flag = np.where(train_lof_scores > lof_threshold, "Anomaly", "Normal")

test_lof_scores = -lof.score_samples(X_test)
test_lof_flag = np.where(test_lof_scores > lof_threshold, "Anomaly", "Normal")

# -------------------------------
# Metrics & overlaps
# -------------------------------

kmeans_avg_sil = silhouette_score(X_train, train_kmeans_labels)
dbscan_avg_sil = dbscan_sil
knn_outliers_count = int((train_knn_outlier == "Outlier").sum())
iso_outliers_count = int((train_iso_flag == "Anomaly").sum())
lof_outliers_count = int((train_lof_flag == "Anomaly").sum())
dbscan_noise = dbscan_noise_count

idx_knn = set(np.where(train_knn_outlier == "Outlier")[0])
idx_iso = set(np.where(train_iso_flag == "Anomaly")[0])
idx_lof = set(np.where(train_lof_flag == "Anomaly")[0])
idx_db_noise = set(np.where(train_db_labels == -1)[0])

overlap_knn_iso = len(idx_knn & idx_iso)
overlap_knn_lof = len(idx_knn & idx_lof)
overlap_knn_db = len(idx_knn & idx_db_noise)
overlap_iso_lof = len(idx_iso & idx_lof)
overlap_iso_db = len(idx_iso & idx_db_noise)
overlap_lof_db = len(idx_lof & idx_db_noise)
overlap_all = len(idx_knn & idx_iso & idx_lof & idx_db_noise)

# Per-species anomaly percentages (train)
def anomaly_pct_per_species(method_flags, positive_value):
    out = {}
    for sp in species_order:
        mask = (y_train == sp)
        total = mask.sum()
        out[sp] = 100.0 * (method_flags[mask] == positive_value).sum() / total if total else 0.0
    return out

pct_knn = anomaly_pct_per_species(train_knn_outlier, "Outlier")
pct_iso = anomaly_pct_per_species(train_iso_flag, "Anomaly")
pct_lof = anomaly_pct_per_species(train_lof_flag, "Anomaly")

# -------------------------------
# Output directory & plots
# -------------------------------

out_dir = ensure_dir("Clustering_Anomaly_Detection_Plots")

# Elbow
plt.figure(figsize=(7, 5))
plt.plot(ks_wss, wss, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Within-Cluster Sum of Squares")
plt.title("Elbow Method for Optimal k")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "elbow_plot.png"), dpi=150)
plt.close()

# Silhouette vs k
plt.figure(figsize=(7, 5))
plt.plot(ks_sil, sil, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Average Silhouette Score")
plt.title("Silhouette Method for Optimal k")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "silhouette_plot.png"), dpi=150)
plt.close()

# KMeans PCA
plt.figure(figsize=(7, 5))
for lab in np.unique(train_kmeans_labels):
    m = (train_kmeans_labels == lab)
    plt.scatter(pca_train[m, 0], pca_train[m, 1], alpha=0.7, label=f"Cluster {lab+1}")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("K-means Clustering (PCA)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "kmeans_plot.png"), dpi=150)
plt.close()

# DBSCAN PCA
plt.figure(figsize=(7, 5))
for lab in np.unique(train_db_labels):
    m = (train_db_labels == lab)
    name = "Noise" if lab == -1 else f"Cluster {lab}"
    plt.scatter(pca_train[m, 0], pca_train[m, 1], alpha=0.7, label=name)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("DBSCAN (PCA)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "dbscan_plot.png"), dpi=150)
plt.close()

# KNN outliers PCA
plt.figure(figsize=(7, 5))
mask_norm = (train_knn_outlier == "Normal")
mask_out = (train_knn_outlier == "Outlier")
plt.scatter(pca_train[mask_norm, 0], pca_train[mask_norm, 1], alpha=0.7, label="Normal")
plt.scatter(pca_train[mask_out, 0], pca_train[mask_out, 1], alpha=0.7, label="Outlier")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("KNN Outlier Detection (PCA)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "knn_plot.png"), dpi=150)
plt.close()

# Isolation Forest PCA
plt.figure(figsize=(7, 5))
mask_norm = (train_iso_flag == "Normal")
mask_anom = (train_iso_flag == "Anomaly")
plt.scatter(pca_train[mask_norm, 0], pca_train[mask_norm, 1], alpha=0.7, label="Normal")
plt.scatter(pca_train[mask_anom, 0], pca_train[mask_anom, 1], alpha=0.7, label="Anomaly")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Isolation Forest (PCA)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "iso_plot.png"), dpi=150)
plt.close()

# LOF PCA
plt.figure(figsize=(7, 5))
mask_norm = (train_lof_flag == "Normal")
mask_anom = (train_lof_flag == "Anomaly")
plt.scatter(pca_train[mask_norm, 0], pca_train[mask_norm, 1], alpha=0.7, label="Normal")
plt.scatter(pca_train[mask_anom, 0], pca_train[mask_anom, 1], alpha=0.7, label="Anomaly")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("LOF (PCA)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "lof_plot.png"), dpi=150)
plt.close()

# Confusion matrices (mapped)
plot_confusion_matrix(kmeans_cm_full, species_order, species_order,
                      "K-Means Confusion Matrix (mapped, TRAIN)",
                      os.path.join(out_dir, "confusion_matrix_kmeans.png"))

if dbscan_cm_full is not None:
    plot_confusion_matrix(dbscan_cm_full, species_order, species_order,
                          "DBSCAN Confusion Matrix (mapped, TRAIN, non-noise)",
                          os.path.join(out_dir, "confusion_matrix_dbscan.png"))

# -------------------------------
# Interactive Plotly toggle (TRAIN)
# -------------------------------

fig = go.Figure()

# Actual species
for sp in species_order:
    m = (y_train == sp)
    fig.add_trace(go.Scatter(x=pca_train[m, 0], y=pca_train[m, 1], mode='markers',
                             name=f"Actual: {sp}", marker=dict(size=8, opacity=0.6), visible=True))

# KMeans
for cl in np.unique(train_kmeans_labels):
    m = (train_kmeans_labels == cl)
    fig.add_trace(go.Scatter(x=pca_train[m, 0], y=pca_train[m, 1], mode='markers',
                             name=f"K-means: C{cl+1}", marker=dict(size=8, opacity=0.6), visible=False))

# DBSCAN
for cl in np.unique(train_db_labels):
    m = (train_db_labels == cl)
    nm = "Noise" if cl == -1 else f"DBSCAN: C{cl}"
    fig.add_trace(go.Scatter(x=pca_train[m, 0], y=pca_train[m, 1], mode='markers',
                             name=nm, marker=dict(size=8, opacity=0.6), visible=False))

# KNN
for lab in ["Normal", "Outlier"]:
    m = (train_knn_outlier == lab)
    fig.add_trace(go.Scatter(x=pca_train[m, 0], y=pca_train[m, 1], mode='markers',
                             name=f"KNN: {lab}", marker=dict(size=8, opacity=0.6), visible=False))

# ISO
for lab in ["Normal", "Anomaly"]:
    m = (train_iso_flag == lab)
    fig.add_trace(go.Scatter(x=pca_train[m, 0], y=pca_train[m, 1], mode='markers',
                             name=f"ISO: {lab}", marker=dict(size=8, opacity=0.6), visible=False))

# LOF
for lab in ["Normal", "Anomaly"]:
    m = (train_lof_flag == lab)
    fig.add_trace(go.Scatter(x=pca_train[m, 0], y=pca_train[m, 1], mode='markers',
                             name=f"LOF: {lab}", marker=dict(size=8, opacity=0.6), visible=False))

n_species = len(species_order)
n_kmeans = len(np.unique(train_kmeans_labels))
n_dbscan = len(np.unique(train_db_labels))
n_knn = 2
n_iso = 2
n_lof = 2

vis_actual = [True]*n_species + [False]*(n_kmeans + n_dbscan + n_knn + n_iso + n_lof)
vis_kmeans = [False]*n_species + [True]*n_kmeans + [False]*(n_dbscan + n_knn + n_iso + n_lof)
vis_dbscan = [False]*(n_species + n_kmeans) + [True]*n_dbscan + [False]*(n_knn + n_iso + n_lof)
vis_knn = [False]*(n_species + n_kmeans + n_dbscan) + [True]*n_knn + [False]*(n_iso + n_lof)
vis_iso = [False]*(n_species + n_kmeans + n_dbscan + n_knn) + [True]*n_iso + [False]*n_lof
vis_lof = [False]*(n_species + n_kmeans + n_dbscan + n_knn + n_iso) + [True]*n_lof

fig.update_layout(
    title="Comparison of Clustering and Anomaly Detection Methods (TRAIN, PCA)",
    xaxis_title="PC1", yaxis_title="PC2",
    updatemenus=[dict(
        type="buttons", direction="right", x=0.1, y=1.15,
        buttons=[
            dict(label="Actual Species", method="update", args=[{"visible": vis_actual}]),
            dict(label="K-means", method="update", args=[{"visible": vis_kmeans}]),
            dict(label="DBSCAN", method="update", args=[{"visible": vis_dbscan}]),
            dict(label="KNN", method="update", args=[{"visible": vis_knn}]),
            dict(label="Isolation Forest", method="update", args=[{"visible": vis_iso}]),
            dict(label="LOF", method="update", args=[{"visible": vis_lof}]),
        ]
    )]
)

pio.write_html(fig, file=os.path.join(out_dir, "interactive_comparison.html"), auto_open=False)

# -------------------------------
# Console summary
# -------------------------------

print("Optimal k based on silhouette score:", optimal_k)
print("Silhouette score on test data:", sil_test)
print("\nK-means clustering accuracy (TRAIN, mapped):", kmeans_acc)
print("K-means Adjusted Rand Index (TRAIN):", kmeans_ari)

print("\nDBSCAN average silhouette score (TRAIN, non-noise):", dbscan_avg_sil)
print("DBSCAN Adjusted Rand Index (TRAIN, excluding noise):", dbscan_ari)
print("Number of DBSCAN noise points (TRAIN):", dbscan_noise)

print("\nEvaluation Metrics:")
print("-------------------")
print("K-means average silhouette score (TRAIN):", kmeans_avg_sil)
print("DBSCAN average silhouette score (TRAIN):", dbscan_avg_sil)
print("Number of KNN outliers (TRAIN):", knn_outliers_count)
print("Number of Isolation Forest anomalies (TRAIN):", iso_outliers_count)
print("Number of LOF anomalies (TRAIN):", lof_outliers_count)
print("Number of DBSCAN noise points (TRAIN):", dbscan_noise)

print("\nOverlap between anomaly detection methods (TRAIN):")
print("----------------------------------------")
print("KNN and Isolation Forest:", overlap_knn_iso, "points")
print("KNN and LOF:", overlap_knn_lof, "points")
print("KNN and DBSCAN noise:", overlap_knn_db, "points")
print("Isolation Forest and LOF:", overlap_iso_lof, "points")
print("Isolation Forest and DBSCAN noise:", overlap_iso_db, "points")
print("LOF and DBSCAN noise:", overlap_lof_db, "points")
print("All four methods:", overlap_all, "points")

print("\nPercentage of each species detected as anomalies (TRAIN):")
print("KNN:", {k: f"{v:.2f}%" for k, v in pct_knn.items()})
print("ISO:", {k: f"{v:.2f}%" for k, v in pct_iso.items()})
print("LOF:", {k: f"{v:.2f}%" for k, v in pct_lof.items()})

print(f"\nSaved figures and interactive plot to: {out_dir}")
