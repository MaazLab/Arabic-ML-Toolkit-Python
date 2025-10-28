# Dataset: Iris
# Output: All plots saved in ./classifications_2/

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelBinarizer

# -------------------------------------------------------------------
# 1. Load dataset
# -------------------------------------------------------------------
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame
df.columns = [c.replace(" (cm)", "") for c in df.columns]  # Clean names
print(df.head())
print(df.describe())

# -------------------------------------------------------------------
# 2. Visualization
# -------------------------------------------------------------------
sns.set(style="whitegrid", palette="deep")

plt.figure()
sns.scatterplot(data=df, x="sepal length", y="sepal width", hue="target")
plt.title("Sepal Length vs Sepal Width")
plt.savefig("classifications_2_sepal_plot.png", dpi=150)

plt.figure()
sns.scatterplot(data=df, x="petal length", y="petal width", hue="target")
plt.title("Petal Length vs Petal Width")
plt.savefig("classifications_2_petal_plot.png", dpi=150)

# -------------------------------------------------------------------
# 3. Preprocessing
# -------------------------------------------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(df.iloc[:, :-1])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# -------------------------------------------------------------------
# 4. KNN Classification
# -------------------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_knn), display_labels=iris.target_names).plot()
plt.title("KNN Confusion Matrix")
plt.savefig("classifications_2_knn_confusion.png", dpi=150)

# -------------------------------------------------------------------
# 5. Linear Regression (multioutput simulation)
# -------------------------------------------------------------------
lb = LabelBinarizer()
Y_train_bin = lb.fit_transform(y_train)

lin_models = []
for i in range(Y_train_bin.shape[1]):
    model = LinearRegression()
    model.fit(X_train, Y_train_bin[:, i])
    lin_models.append(model)

Y_pred_scores = np.column_stack([m.predict(X_test) for m in lin_models])
y_pred_lin = Y_pred_scores.argmax(axis=1)
acc_lin = accuracy_score(y_test, y_pred_lin)

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_lin), display_labels=iris.target_names).plot()
plt.title("Linear Regression Confusion Matrix")
plt.savefig("classifications_2_linear_confusion.png", dpi=150)

# -------------------------------------------------------------------
# 6. Decision Tree
# -------------------------------------------------------------------
dt = DecisionTreeClassifier(random_state=123)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)

plt.figure(figsize=(10, 6))
plot_tree(dt, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Decision Tree Visualization")
plt.savefig("classifications_2_decision_tree.png", dpi=150)

# Variable importance
feat_imp = pd.Series(dt.feature_importances_, index=iris.feature_names).sort_values(ascending=True)
plt.figure()
feat_imp.plot(kind="barh", color="steelblue")
plt.title("Decision Tree Feature Importance")
plt.savefig("classifications_2_decision_tree_importance.png", dpi=150)

# -------------------------------------------------------------------
# 7. Logistic Regression (multinomial)
# -------------------------------------------------------------------
log_reg = LogisticRegression(max_iter=500, multi_class='multinomial', solver='lbfgs')
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)
acc_log = accuracy_score(y_test, y_pred_log)

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_log), display_labels=iris.target_names).plot()
plt.title("Logistic Regression Confusion Matrix")
plt.savefig("classifications_2_logistic_confusion.png", dpi=150)

# -------------------------------------------------------------------
# 8. Model Comparison
# -------------------------------------------------------------------
comparison = pd.DataFrame({
    "Model": ["KNN", "Linear Regression", "Decision Tree", "Logistic Regression"],
    "Accuracy": [acc_knn, acc_lin, acc_dt, acc_log]
}).sort_values(by="Accuracy")

plt.figure()
sns.barplot(data=comparison, x="Accuracy", y="Model", palette="Blues_d")
plt.title("Model Comparison (Accuracy)")
plt.xlim(0, 1)
plt.savefig("classifications_2_model_comparison.png", dpi=150)

# -------------------------------------------------------------------
# 9. Decision Boundary for KNN (using Petal features only)
# -------------------------------------------------------------------
def plot_knn_boundary():
    X_petals = df[["petal length", "petal width"]].values
    y_labels = df["target"]
    knn_small = KNeighborsClassifier(n_neighbors=5).fit(X_petals, y_labels)

    x_min, x_max = X_petals[:, 0].min() - 0.5, X_petals[:, 0].max() + 0.5
    y_min, y_max = X_petals[:, 1].min() - 0.5, X_petals[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
    Z = knn_small.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
    sns.scatterplot(x=X_petals[:, 0], y=X_petals[:, 1], hue=iris.target_names[y_labels])
    plt.title("KNN Decision Boundaries (k=5)")
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.savefig("classifications_2_knn_decision_boundary.png", dpi=150)

plot_knn_boundary()

# -------------------------------------------------------------------
# 10. Save all plots in directory
# -------------------------------------------------------------------
os.makedirs("classifications_2", exist_ok=True)
print("✅ All classification models executed successfully.")
print("🖼️ Plots saved inside ./classifications_2/")
