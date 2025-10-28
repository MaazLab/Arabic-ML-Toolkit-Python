## 📊 **Model Performance Summary**

### 🧠 Task Overview

The classification task aimed to **distinguish between nonspam and spam messages** using **Support Vector Machines (SVM)** and **Bernoulli Naive Bayes** models.
The experiment demonstrates **strong model performance** with effective spam detection and low false-positive rates.

---

### 📚 **Dataset Description**

| Attribute            | Details                                                                                                                                                                                                     |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Dataset Name**     | [Spambase Dataset](https://www.openml.org/d/44)                                                                                                                                                             |
| **Source**           | UCI Machine Learning Repository (via OpenML)                                                                                                                                                                |
| **Size**             | 4,601 samples × 58 features                                                                                                                                                                                 |
| **Target Variable**  | `class` — 1 = spam, 0 = nonspam                                                                                                                                                                             |
| **Feature Types**    | 57 continuous features + 1 binary target                                                                                                                                                                    |
| **Domain**           | Email spam filtering                                                                                                                                                                                        |
| **Language**         | English                                                                                                                                                                                                     |
| **Description**      | Each observation represents an email, with features representing word or character frequency (e.g., “make”, “address”, “free”, “money”), and statistical measures like average/longest capital letter runs. |
| **Example Features** | `word_freq_free`, `word_freq_money`, `char_freq_$`, `capital_run_length_average`                                                                                                                            |

> 💡 *The dataset is widely used for text classification and spam detection benchmarking.
> In this project, features were binarized (1 if frequency > 0, else 0) for compatibility with Bernoulli Naive Bayes.*

---

### ✅ **Key Metrics (Bernoulli Naive Bayes)**

| Metric        | R Implementation | Python Implementation | Interpretation                                   |
| :------------ | :--------------: | :-------------------: | :----------------------------------------------- |
| **Accuracy**  |      82.74%      |       **89.50%**      | Correctly classified the majority of messages.   |
| **AUC (ROC)** |      0.9585      |       **0.9506**      | Excellent separability between spam and nonspam. |
| **F1 Score**  |      0.8383      |       **0.8620**      | Balanced precision and recall.                   |
| **Kappa**     |      0.6604      |           –           | Substantial agreement beyond chance.             |

---

### 📌 **Class-wise Performance**

#### **Nonspam (Positive Class)**

* **R (Sensitivity / Recall):** 0.7380
* **Python (Recall):** 0.9355
  → The Python implementation improved the recall, meaning fewer nonspam messages were misclassified.
* **NPV (R):** 0.7052

#### **Spam**

* **R (Specificity):** 0.9650
* **Python (Recall):** 0.8327
* **Precision (R):** 0.9701
* **Precision (Python):** 0.8935

---

### 📈 **Highlights**

* **High AUC and precision** demonstrate excellent ability to separate spam from legitimate messages.
* **Python version outperforms the R version** in overall accuracy and nonspam recall, likely due to better data handling and binarization consistency.
* Both versions show **robust spam detection**, maintaining a good balance between sensitivity and specificity.
* The **Bernoulli Naive Bayes** classifier is efficient, interpretable, and suitable as a baseline for further enhancements.

---

### 🧾 **Conclusion**

The **Spambase dataset** experiment shows that **Bernoulli Naive Bayes** is a reliable model for email spam filtering, achieving **~90% accuracy** and **AUC ≈ 0.95**.
Its simplicity and performance make it ideal for rapid experimentation and educational demonstrations within the **Arabic ML Toolkit** project.

