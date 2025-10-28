# 🔍 Clustering & Anomaly Detection Summary (Iris Dataset)

## 📘 Dataset Overview

**Dataset:** Iris (Fisher’s Iris Flower Dataset)  
**Source:** Built-in dataset from `sklearn.datasets.load_iris`  
**Samples:** 150 observations  
**Features:** 4 continuous numeric attributes  
- 🌿 *Sepal Length* (cm)  
- 🌿 *Sepal Width* (cm)  
- 🌸 *Petal Length* (cm)  
- 🌸 *Petal Width* (cm)  

**Target Classes (Species):**
- *Setosa* (50 samples)
- *Versicolor* (50 samples)
- *Virginica* (50 samples)

**Objective:**  
Perform unsupervised clustering (K-Means, DBSCAN) and anomaly detection (KNN Distance, Isolation Forest, LOF) to explore natural groupings and detect outlier points in the dataset.

---

## 📊 Clustering Results

### ✅ K-Means Clustering
- **Optimal Clusters (k):** 2 (determined by silhouette score)
- **Average Silhouette Score:** **0.65** (good separation)
- **Accuracy (mapped to species):** **67.2%**
- **Adjusted Rand Index (ARI):** **0.579** (moderate alignment)

**Observation:**  
K-Means clearly separated *Setosa* as one group and combined *Versicolor* and *Virginica* into another.  
This reflects their natural overlap in petal size and shape, which often confuses linear partitioning methods.

---

### ✅ DBSCAN Clustering
- **Suggested `eps` value:** **0.395**
- **`minPts`:** 5 (based on number of features + 1)
- **Clusters Identified:** 4 (including one major noise group)
- **Noise Points:** 68 (≈45% of total samples)
- **Average Silhouette Score:** **0.38**
- **Adjusted Rand Index (ARI):** **0.356**

**Observation:**  
DBSCAN effectively isolated *Setosa*, but high sensitivity to parameters caused many *Versicolor* and *Virginica* samples to be labeled as noise.  
While it identifies dense regions, the overlapping feature space of these species limited performance.

---

## ⚠️ Anomaly Detection Results

| Method              | Outliers Detected |
|---------------------|------------------:|
| **KNN Distance**    | 7 |
| **Isolation Forest**| 7 |
| **LOF (Local Outlier Factor)** | 7 |
| **DBSCAN Noise Points** | 68 |

### 🔄 Overlap Among Methods
- **All four methods agreed on:** 5 anomalies  
- Strong overlap between **KNN**, **Isolation Forest**, and **LOF**, confirming consistent anomaly detection.

---

### 📈 Anomalies by Species

| Method | Setosa (%) | Versicolor (%) | Virginica (%) |
|:--------|------------:|---------------:|---------------:|
| **KNN Distance** | 4.76% | 2.50% | 10.00% |
| **Isolation Forest** | 7.14% | 2.50% | 7.50% |
| **LOF** | 7.14% | 2.50% | 7.50% |

**Observation:**
- *Virginica* shows the highest anomaly frequency, suggesting greater variability or boundary overlap.
- *Setosa* remains the most distinct and stable species.
- Consistent detection across multiple methods reinforces the reliability of anomaly identification.

---

## 🧠 Conclusions

1. **K-Means**
   - Performs well for well-separated clusters like *Setosa*.
   - Tends to merge overlapping classes (*Versicolor*, *Virginica*).
   - Best suited for spherical clusters with similar variance.

2. **DBSCAN**
   - Excels at noise detection and non-linear boundaries.
   - Requires careful tuning of `eps` and `minPts`.
   - In this dataset, overestimated noise due to overlapping regions.

3. **Anomaly Detection**
   - **KNN Distance**, **Isolation Forest**, and **LOF** agree on key anomalies.
   - Indicates *Virginica* contains samples that deviate more from the core distribution.

4. **Overall Insight**
   - Combining multiple unsupervised algorithms provides complementary perspectives:
     - **Clustering** → reveals structure.
     - **Anomaly Detection** → validates and explains boundary cases.
   - This ensemble approach enhances reliability when no ground truth is available.

---

### 📁 Output Files Generated

All plots and outputs are saved under:

Clustering_Anomaly_Detection_Plots/
├── elbow_plot.png
├── silhouette_plot.png
├── kmeans_plot.png
├── dbscan_plot.png
├── knn_plot.png
├── iso_plot.png
├── lof_plot.png
├── confusion_matrix_kmeans.png
├── confusion_matrix_dbscan.png
└── interactive_comparison.html

---

**Developed in Python (Scikit-Learn + Plotly)**  
Implements the full pipeline from preprocessing → clustering → anomaly detection → visualization → report generation.
