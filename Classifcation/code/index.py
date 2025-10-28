# ============================================================
# Bernoulli Naive Bayes Classification (Python Equivalent of R Code)
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc
)

# ------------------------------------------------------------
# 1. Load dataset (same "spam" dataset from kernlab)
# ------------------------------------------------------------
print("📦 Loading dataset...")
spam = fetch_openml(name="spambase", version=1, as_frame=True)
df = spam.frame

print(f"Dataset shape: {df.shape}")
print(df.head())

# ------------------------------------------------------------
# 2. Preprocess — binarize numeric features (>0 → 1)
# ------------------------------------------------------------
X = df.drop(columns=["class"])
y = pd.to_numeric(df["class"]).map({1: "spam", 0: "nonspam"})

print(f'df["class"] unique values: {df["class"].unique()}')
print("y.value_counts(): ", y.value_counts())
print("y.isna().sum(): ", y.isna().sum())  # should be 0

# Bernoulli requires binary features
X_bin = (X > 0).astype(int)

# ------------------------------------------------------------
# 3. Split data (70% train, 30% test)
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_bin, y, test_size=0.3, random_state=123, stratify=y
)

# ------------------------------------------------------------
# 4. Train Bernoulli Naive Bayes model
# ------------------------------------------------------------
model = BernoulliNB()
model.fit(X_train, y_train)

# ------------------------------------------------------------
# 5. Predictions
# ------------------------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ------------------------------------------------------------
# 6. Evaluation
# ------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred, labels=["nonspam", "spam"])
report = classification_report(y_test, y_pred, target_names=["nonspam", "spam"], output_dict=True)
auc_score = roc_auc_score((y_test == "spam").astype(int), y_prob)

print("\n=== Confusion Matrix ===")
print(pd.DataFrame(cm, index=["Actual Nonspam", "Actual Spam"], columns=["Pred Nonspam", "Pred Spam"]))
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))
print(f"AUC: {auc_score:.4f}")

# ------------------------------------------------------------
# 7. Visualization directory
# ------------------------------------------------------------
output_dir = "classification_plots"
os.makedirs(output_dir, exist_ok=True)

# ------------------------------------------------------------
# 8. Confusion Matrix Plot
# ------------------------------------------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Nonspam", "Spam"], yticklabels=["Nonspam", "Spam"])
plt.title("Confusion Matrix - Bernoulli Naive Bayes")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 9. ROC Curve
# ------------------------------------------------------------
fpr, tpr, _ = roc_curve((y_test == "spam").astype(int), y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], color="red", linestyle="--")
plt.title("ROC Curve - Bernoulli Naive Bayes")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{output_dir}/roc_curve.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 10. Feature Importance (Correlation)
# ------------------------------------------------------------
# Convert target to numeric (spam=1, nonspam=0)
y_num = (y_train == "spam").astype(int)
correlations = X_train.corrwith(y_num).abs().sort_values(ascending=False)

top_features = correlations.head(15)
plt.figure(figsize=(8, 6))
sns.barplot(x=top_features.values, y=top_features.index, color="steelblue")
plt.title("Top 15 Features by Absolute Correlation with Target")
plt.xlabel("Absolute Correlation")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(f"{output_dir}/top15_features.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 11. Probability Distribution Plot
# ------------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.kdeplot(x=y_prob[y_test == "spam"], fill=True, color="red", label="Spam", alpha=0.5)
sns.kdeplot(x=y_prob[y_test == "nonspam"], fill=True, color="green", label="Nonspam", alpha=0.5)
plt.title("Distribution of Predicted Probabilities by Class")
plt.xlabel("Predicted Probability of Spam")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/probability_distribution.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 12. Precision–Recall Curve
# ------------------------------------------------------------
precision, recall, _ = precision_recall_curve((y_test == "spam").astype(int), y_prob)
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, color="darkorange", lw=2)
plt.title("Precision–Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.savefig(f"{output_dir}/precision_recall_curve.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 13. Prediction Confidence Histogram
# ------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.hist(y_prob, bins=np.arange(0, 1.05, 0.05), color="skyblue", edgecolor="black")
plt.title("Histogram of Spam Prediction Confidence")
plt.xlabel("Probability of Spam")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(f"{output_dir}/prediction_confidence_histogram.png", dpi=300)
plt.close()

# ------------------------------------------------------------
# 14. Final Report
# ------------------------------------------------------------
print("\n📊 Classification Report Summary:")
print(f"Accuracy: {report['accuracy']:.4f}")
print(f"Recall (nonspam): {report['nonspam']['recall']:.4f}")
print(f"Recall (spam): {report['spam']['recall']:.4f}")
print(f"Precision (spam): {report['spam']['precision']:.4f}")
f1_score = report['spam']['f1-score']
print(f"F1 Score: {f1_score:.4f}")
print(f"AUC: {roc_auc:.4f}")
print(f"✅ All plots saved to: {output_dir}/")
