# ======================================================
# Semi-Supervised Learning Methods Comparison
# Self-Training | Co-Training | Label Propagation
# ======================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.semi_supervised import LabelPropagation
from sklearn.utils import resample
from sklearn.datasets import fetch_openml

from scipy.stats import mode
import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(42)


def save_plot(fig, filename, folder="plots"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot: {path}")

# ======================================================
# 1. Dataset Preparation
# ======================================================

def load_dataset(name):
    if name == "iris":
        data = load_iris(as_frame=True)
        X = data.data
        # Map numeric labels (0,1,2) → species names directly
        y = pd.Series(pd.Categorical.from_codes(data.target, categories=data.target_names))
        return X, y
    elif name == "pima":
        # Fetch the Pima Indians Diabetes dataset safely by ID
        df = fetch_openml(data_id=37, as_frame=True).frame
        # Standardize column naming across OpenML versions
        target_col = "class" if "class" in df.columns else "Class"
        X = df.drop(columns=[target_col])
        y = df[target_col].astype("category")
        return X, y
    else:
        raise ValueError("Dataset not supported. Use 'iris' or 'pima'.")


def stratified_split(X, y, label_fraction=0.1, test_fraction=0.2):
    # Split test set first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, stratify=y, random_state=42
    )
    # Split labeled/unlabeled
    X_labeled, X_unlabeled, y_labeled, y_unlabeled = [], [], [], []
    for cls in y_train.unique():
        X_c = X_train[y_train == cls]
        y_c = y_train[y_train == cls]
        n_lab = int(len(X_c) * label_fraction)
        X_labeled.append(X_c.iloc[:n_lab])
        y_labeled.append(y_c.iloc[:n_lab])
        X_unlabeled.append(X_c.iloc[n_lab:])
        y_unlabeled.append(y_c.iloc[n_lab:])
    return (
        pd.concat(X_labeled),
        pd.concat(X_unlabeled),
        pd.concat(y_labeled),
        pd.concat(y_unlabeled),
        X_test,
        y_test,
    )

# ======================================================
# 2. Semi-Supervised Methods
# ======================================================

def self_training(X_lab, y_lab, X_unlab, y_unlab, X_test, y_test,
                  max_iter=10, k_per_iter=10, threshold=0.8):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_lab, y_lab)

    acc_hist = [accuracy_score(y_test, model.predict(X_test))]
    pseudo_acc = []

    unlabeled_idx = np.arange(len(X_unlab))
    X_u = X_unlab.copy()
    y_true = y_unlab.copy()

    for it in range(max_iter):
        probs = model.predict_proba(X_u)
        conf = probs.max(axis=1)
        preds = model.classes_[probs.argmax(axis=1)]
        confident = np.where(conf >= threshold)[0]

        if len(confident) == 0:
            threshold *= 0.9
            if threshold < 0.5:
                break
            continue

        top_k = confident[np.argsort(conf[confident])[-k_per_iter:]]
        X_add = X_u.iloc[top_k]
        y_add = preds[top_k]
        pseudo_acc.extend((y_add == y_true.iloc[top_k]).tolist())

        # Add to labeled set
        X_lab = pd.concat([X_lab, X_add])
        y_lab = pd.concat([y_lab, pd.Series(y_add, index=X_add.index)])

        # Remove from unlabeled
        X_u = X_u.drop(index=X_add.index)
        y_true = y_true.drop(index=X_add.index)

        # Retrain
        model.fit(X_lab, y_lab)
        acc = accuracy_score(y_test, model.predict(X_test))
        acc_hist.append(acc)
        print(f"[Self-Training] Iter {it+1}: added {len(X_add)} → acc={acc:.3f}")

        if X_u.empty:
            break

    pseudo_label_acc = np.mean(pseudo_acc) if pseudo_acc else np.nan
    return model, acc_hist, pseudo_label_acc


def co_training(X_lab, y_lab, X_unlab, y_unlab, X_test, y_test,
                max_iter=10, k_per_iter=10, threshold=0.8):
    # Split features into two views (random)
    features = np.array(X_lab.columns) 
    np.random.shuffle(features)
    mid = len(features) // 2
    view1, view2 = features[:mid], features[mid:]

    m1 = RandomForestClassifier(n_estimators=100, random_state=42)
    m2 = RandomForestClassifier(n_estimators=100, random_state=42)
    m1.fit(X_lab[view1], y_lab)
    m2.fit(X_lab[view2], y_lab)

    pred1 = m1.predict(X_test[view1])
    pred2 = m2.predict(X_test[view2])
    stacked = np.vstack([pred1, pred2])
    ensemble_preds = np.array([np.unique(col)[np.argmax(np.bincount(pd.factorize(col)[0]))] for col in stacked.T])
    acc_hist = [accuracy_score(y_test, ensemble_preds)]
    pseudo_acc = []

    X_u, y_true = X_unlab.copy(), y_unlab.copy()

    for it in range(max_iter):
        probs1 = m1.predict_proba(X_u[view1])
        probs2 = m2.predict_proba(X_u[view2])

        conf1, preds1 = probs1.max(axis=1), m1.classes_[probs1.argmax(axis=1)]
        conf2, preds2 = probs2.max(axis=1), m2.classes_[probs2.argmax(axis=1)]

        idx1 = np.where(conf1 >= threshold)[0]
        idx2 = np.where(conf2 >= threshold)[0]

        idx = np.unique(np.concatenate([idx1[:k_per_iter//2], idx2[:k_per_iter//2]]))
        if len(idx) == 0:
            threshold *= 0.9
            if threshold < 0.5:
                break
            continue

        y_add = []
        for i in idx:
            if i in idx1 and i in idx2:
                y_add.append(preds1[i] if conf1[i] >= conf2[i] else preds2[i])
            elif i in idx1:
                y_add.append(preds1[i])
            else:
                y_add.append(preds2[i])

        X_add = X_u.iloc[idx]
        pseudo_acc.extend((np.array(y_add) == y_true.iloc[idx]).tolist())

        X_lab = pd.concat([X_lab, X_add])
        y_lab = pd.concat([y_lab, pd.Series(y_add, index=X_add.index)])
        X_u = X_u.drop(index=X_add.index)
        y_true = y_true.drop(index=X_add.index)

        m1.fit(X_lab[view1], y_lab)
        m2.fit(X_lab[view2], y_lab)

        pred1 = m1.predict_proba(X_test[view1])
        pred2 = m2.predict_proba(X_test[view2])
        combined = (pred1 + pred2) / 2
        preds = m1.classes_[combined.argmax(axis=1)]
        acc = accuracy_score(y_test, preds)
        acc_hist.append(acc)
        print(f"[Co-Training] Iter {it+1}: added {len(idx)} → acc={acc:.3f}")

        if X_u.empty:
            break

    pseudo_label_acc = np.mean(pseudo_acc) if pseudo_acc else np.nan
    return (m1, m2), acc_hist, pseudo_label_acc


def label_propagation_method(X_lab, y_lab, X_unlab, y_unlab, X_test, y_test):
    # Encode string labels → numeric codes
    all_labels = pd.concat([y_lab, y_unlab])
    y_codes, uniques = pd.factorize(all_labels)

    # Map encoded labels
    y_lab_encoded = pd.Series(y_codes[:len(y_lab)], index=y_lab.index)
    y_unlab_encoded = pd.Series(y_codes[len(y_lab):], index=y_unlab.index)

    # Prepare combined labels with -1 for unlabeled
    y_combined = pd.concat([y_lab_encoded, pd.Series([-1]*len(X_unlab), index=X_unlab.index)])

    X_combined = pd.concat([X_lab, X_unlab])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    # Train label propagation model
    lp = LabelPropagation(kernel="rbf", gamma=0.5, max_iter=1000)
    lp.fit(X_scaled, y_combined)

    # Get propagated pseudo labels (decoded back)
    propagated = uniques[lp.transduction_[-len(X_unlab):]]
    pseudo_acc = accuracy_score(y_unlab, propagated)

    # Final model trained on all pseudo-labeled data
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    y_final = pd.concat([y_lab, pd.Series(propagated, index=X_unlab.index)])
    model.fit(X_combined, y_final)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"[Label Propagation] acc={acc:.3f}, pseudo_acc={pseudo_acc:.3f}")

    return model, acc, pseudo_acc



# ======================================================
# 3. Visualization & Comparison
# ======================================================

def plot_accuracy(histories, dataset):
    plt.figure(figsize=(8,5))
    for name, h in histories.items():
        plt.plot(range(len(h)), h, marker='o', label=name)
    plt.title(f"Performance Comparison on {dataset}")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    
    # Save and show
    fig = plt.gcf()
    save_plot(fig, f"{dataset}_accuracy_trend.png")
    plt.close(fig)

# ======================================================
# 4. Run Experiments
# ======================================================

def run_experiments(dataset_name="iris"):
    X, y = load_dataset(dataset_name)
    X_lab, X_unlab, y_lab, y_unlab, X_test, y_test = stratified_split(X, y)

    print(f"\nDataset: {dataset_name}")
    print(f"Labeled: {len(X_lab)}, Unlabeled: {len(X_unlab)}, Test: {len(X_test)}")

    # Supervised baseline
    base = RandomForestClassifier(n_estimators=100, random_state=42)
    base.fit(X_lab, y_lab)
    base_acc = accuracy_score(y_test, base.predict(X_test))
    print(f"Supervised Baseline acc={base_acc:.3f}")

    # Self-Training
    m_self, h_self, acc_self_pseudo = self_training(X_lab, y_lab, X_unlab, y_unlab, X_test, y_test)
    # Co-Training
    m_co, h_co, acc_co_pseudo = co_training(X_lab, y_lab, X_unlab, y_unlab, X_test, y_test)
    # Label Propagation
    m_lp, acc_lp, acc_lp_pseudo = label_propagation_method(X_lab, y_lab, X_unlab, y_unlab, X_test, y_test)

    histories = {
        "Supervised": [base_acc]*len(h_self),
        "Self-Training": h_self,
        "Co-Training": h_co
    }
    plot_accuracy(histories, dataset_name)

    summary = pd.DataFrame({
        "Method": ["Supervised", "Self-Training", "Co-Training", "Label Propagation"],
        "Accuracy": [base_acc, h_self[-1], h_co[-1], acc_lp],
        "PseudoLabelAcc": [np.nan, acc_self_pseudo, acc_co_pseudo, acc_lp_pseudo]
    })
    print("\nSummary:")
    print(summary)
    return summary

def save_summary_plots(results_dict):
    """
    results_dict: {'iris': iris_summary, 'pima': pima_summary}
    """
    os.makedirs("plots", exist_ok=True)
    summary_all = []
    for dataset, df in results_dict.items():
        df["Dataset"] = dataset
        summary_all.append(df)
    summary_all = pd.concat(summary_all, ignore_index=True)

    # Convert accuracy to percentage
    summary_all["Accuracy(%)"] = summary_all["Accuracy"] * 100
    summary_all["PseudoLabelAcc(%)"] = summary_all["PseudoLabelAcc"] * 100

    # Plot 1: Accuracy comparison
    plt.figure(figsize=(10,6))
    sns.barplot(x="Method", y="Accuracy(%)", hue="Dataset", data=summary_all)
    plt.title("Accuracy Comparison Across Datasets")
    plt.xlabel("Method")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    save_plot(plt.gcf(), "accuracy_comparison.png")
    plt.close()

    # Plot 2: Pseudo-label quality
    plt.figure(figsize=(10,6))
    sns.barplot(x="Method", y="PseudoLabelAcc(%)", hue="Dataset", data=summary_all)
    plt.title("Pseudo-Label Accuracy Across Datasets")
    plt.xlabel("Method")
    plt.ylabel("Pseudo-Label Accuracy (%)")
    plt.xticks(rotation=30)
    plt.tight_layout()
    save_plot(plt.gcf(), "pseudo_label_comparison.png")
    plt.close()

    print("✅ All summary plots saved in 'plots/' folder.")


if __name__ == "__main__":
    iris_summary = run_experiments("iris")
    pima_summary = run_experiments("pima")
    
    all_results = {"iris": iris_summary, "pima": pima_summary}
    save_summary_plots(all_results)
