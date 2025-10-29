# 🧠 Semi-Supervised Learning Summary – *Iris & PIMA Datasets (Python Implementation)*

---

## 📊 Final Results Overview

| Dataset  | Method              | Accuracy | Pseudo-Label Accuracy |
| -------- | ------------------- | -------- | --------------------- |
| **iris** | Supervised Baseline | 86.67%   | –                     |
| **iris** | Self-Training       | 96.67%   | 98.97%                |
| **iris** | Co-Training         | 83.33%   | 100.00%               |
| **iris** | Label Propagation   | 66.67%   | 68.52%                |
| **pima** | Supervised Baseline | 74.51%   | –                     |
| **pima** | Self-Training       | 76.47%   | 80.10%                |
| **pima** | Co-Training         | 72.55%   | 78.23%                |
| **pima** | Label Propagation   | 75.49%   | 70.18%                |

---

## 🌸 **IRIS Dataset Insights**

* **Baseline Accuracy (Supervised with 10% labeled data):** 86.67%
  The model performs well even with limited labeled examples due to the simplicity and separability of Iris features.
* **Self-Training:** Achieved **96.67% accuracy**, showing strong iterative learning and confident pseudo-labeling.
* **Co-Training:** Stable but slightly lower performance (**83.33%**) — affected by feature view splits and noisy co-labeling.
* **Label Propagation:** Reached **66.67%**, reflecting limited success on multi-class separation.

📈 *Observation:*
Self-Training clearly dominated here by leveraging high-confidence predictions to expand the labeled dataset effectively.

✅ **Best Semi-Supervised Method:** Self-Training
❌ **Least Effective:** Label Propagation

---

## 🧬 **PIMA Diabetes Dataset Insights**

* **Baseline Accuracy (10% labeled data):** 74.51%
* **Self-Training:** Slight improvement to **76.47%**, indicating moderate benefit from pseudo-labeled data.
* **Co-Training:** Performed comparably (**72.55%**) but sensitive to view partitioning.
* **Label Propagation:** Achieved **75.49%**, showing stability though limited enhancement.

📈 *Observation:*
Performance variations were small — the unlabeled data added marginal value, possibly due to overlapping feature space and noisy medical attributes.

✅ **Best Semi-Supervised Method:** Self-Training (by a small margin)
⚠️ **Co-Training and Label Propagation:** Performed comparably to baseline without significant gains.

---

## 🔎 **General Conclusions**

* **Self-Training** is the most robust and consistently improving semi-supervised method across datasets.
* **Co-Training** depends heavily on how features are split into independent “views.”
* **Label Propagation** requires clear class boundaries and performs better in low-dimensional, binary problems.
* Semi-supervised methods are most effective when:

  * Labeled data is scarce but representative,
  * Unlabeled data comes from the same distribution,
  * The model avoids reinforcing early labeling errors.

---

✅ *This report corresponds to the Python implementation of the Semi-Supervised Learning Toolkit (Self-Training, Co-Training, Label Propagation) applied to the Iris and PIMA datasets. All plots have been saved in the `/plots` directory.*
