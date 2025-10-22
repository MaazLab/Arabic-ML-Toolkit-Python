# Install required packages if not already installed
if (!require("kernlab")) install.packages("kernlab")
if (!require("caret")) install.packages("caret")
if (!require("e1071")) install.packages("e1071")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("pROC")) install.packages("pROC")

# Load libraries
library(kernlab)
library(caret)
library(e1071)
library(ggplot2)
library(pROC)

# Load dataset
data("spam")

# Examine the data
head(spam)
str(spam)
summary(spam)

# Preprocess the data
# For Bernoulli Naive Bayes, we need binary features
# We'll convert numeric features to binary based on whether they're > 0

# Create a function to binarize features
binarize <- function(x) {
  return(as.numeric(x > 0))
}

# Apply binarization to all numeric columns except the target
spam_binary <- as.data.frame(lapply(spam[, -58], binarize))

# Add back the target variable
spam_binary$type <- spam$type

# Split the data into training and testing sets (70% train, 30% test)
set.seed(123)  # For reproducibility
trainIndex <- createDataPartition(spam_binary$type, p = 0.7, list = FALSE)
train_data <- spam_binary[trainIndex, ]
test_data <- spam_binary[-trainIndex, ]

# Train the Bernoulli Naive Bayes model
bernoulli_nb_model <- naiveBayes(type ~ ., data = train_data)

# Make predictions on the test set
predictions <- predict(bernoulli_nb_model, test_data[, -ncol(test_data)])
prob_predictions <- predict(bernoulli_nb_model, test_data[, -ncol(test_data)], type = "raw")

# Evaluate the model
conf_matrix <- confusionMatrix(predictions, test_data$type)
print(conf_matrix)

# Calculate AUC-ROC
roc_obj <- roc(as.numeric(test_data$type), prob_predictions[, 2])
auc_value <- auc(roc_obj)
print(paste("AUC:", round(auc_value, 4)))

# Visualizations

# 1. Confusion Matrix Visualization
cm_table <- conf_matrix$table
cm_data <- as.data.frame(as.table(cm_table))
names(cm_data) <- c("Predicted", "Actual", "Freq")

ggplot(cm_data, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "black", size = 5) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Confusion Matrix for Bernoulli Naive Bayes",
       x = "Actual Class",
       y = "Predicted Class") +
  theme_minimal()

# 2. ROC Curve
plot(roc_obj, main = "ROC Curve for Bernoulli Naive Bayes", 
     col = "blue", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "red")
text(0.7, 0.3, paste("AUC =", round(auc_value, 4)), col = "black")

# 3. Simple Correlation-based Feature Importance 
# This avoids the error by using a different approach altogether
feature_correlations <- data.frame(
  Feature = character(),
  Correlation = numeric(),
  stringsAsFactors = FALSE
)

# Calculate correlation of each feature with the target
for (feature in names(train_data)[-ncol(train_data)]) {
  # Convert factor to numeric (nonspam = 0, spam = 1)
  target_numeric <- as.numeric(train_data$type) - 1
  
  # Calculate point-biserial correlation (correlation between binary features and binary target)
  correlation <- abs(cor(train_data[[feature]], target_numeric))
  
  # Add to results
  feature_correlations <- rbind(feature_correlations, 
                               data.frame(Feature = feature, 
                                          Correlation = correlation))
}

# Sort by correlation (higher absolute correlation = more important)
feature_correlations <- feature_correlations[order(-feature_correlations$Correlation), ]

# Plot top 15 features
ggplot(head(feature_correlations, 15), aes(x = reorder(Feature, Correlation), y = Correlation)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 15 Important Features by Correlation with Target",
       x = "Feature",
       y = "Absolute Correlation with Target") +
  theme_minimal()

# 4. Probability Distribution Plot
pred_data <- data.frame(
  Actual = test_data$type,
  Probability = prob_predictions[, 2]
)

ggplot(pred_data, aes(x = Probability, fill = Actual)) +
  geom_density(alpha = 0.5) +
  labs(title = "Distribution of Predicted Probabilities by Class",
       x = "Predicted Probability of Spam",
       y = "Density") +
  scale_fill_manual(values = c("green", "red")) +
  theme_minimal()

# Print classification report
cat("\nClassification Report:\n")
cat("Accuracy:", round(conf_matrix$overall['Accuracy'], 4), "\n")
cat("Sensitivity (Recall for nonspam):", round(conf_matrix$byClass['Sensitivity'], 4), "\n")
cat("Specificity (Recall for spam):", round(conf_matrix$byClass['Specificity'], 4), "\n")
cat("Precision (Positive Predictive Value):", round(conf_matrix$byClass['Pos Pred Value'], 4), "\n")
cat("F1 Score:", round(2 * conf_matrix$byClass['Sensitivity'] * conf_matrix$byClass['Pos Pred Value'] / 
                   (conf_matrix$byClass['Sensitivity'] + conf_matrix$byClass['Pos Pred Value']), 4), "\n")
cat("AUC:", round(auc_value, 4), "\n")


# Save the plots
# Create output directory if it doesn't exist
output_dir <- "classfication_plots"
if (!dir.exists(output_dir)) dir.create(output_dir)

# 1. Confusion Matrix Plot
cm_plot <- ggplot(cm_data, aes(x = Actual, y = Predicted, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "black", size = 5) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Confusion Matrix for Bernoulli Naive Bayes",
       x = "Actual Class", y = "Predicted Class") +
  theme_minimal()
ggsave(file.path(output_dir, "confusion_matrix.png"), cm_plot, width = 7, height = 5, dpi = 300)

# 2. ROC Curve
png(file.path(output_dir, "roc_curve.png"), width = 700, height = 500)
plot(roc_obj, main = "ROC Curve for Bernoulli Naive Bayes", col = "blue", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "red")
text(0.7, 0.3, paste("AUC =", round(auc_value, 4)), col = "black")
dev.off()

# 3. Feature Importance Plot
feat_plot <- ggplot(head(feature_correlations, 15), aes(x = reorder(Feature, Correlation), y = Correlation)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 15 Important Features by Correlation with Target",
       x = "Feature", y = "Abs Correlation with Target") +
  theme_minimal()
ggsave(file.path(output_dir, "top15_features.png"), feat_plot, width = 8, height = 6, dpi = 300)


# 5. Precision-Recall Curve
precision <- roc_obj$sensitivities
recall <- roc_obj$specificities
pr_data <- data.frame(Precision = precision, Recall = recall)
pr_plot <- ggplot(pr_data, aes(x = Recall, y = Precision)) +
  geom_line(color = "darkorange", size = 1.5) +
  labs(title = "Precision-Recall Curve", x = "Recall", y = "Precision") +
  theme_minimal()
ggsave(file.path(output_dir, "precision_recall_curve.png"), pr_plot, width = 6, height = 5, dpi = 300)

# 6. Prediction Confidence Histogram
conf_plot <- ggplot(pred_data, aes(x = Probability)) +
  geom_histogram(binwidth = 0.05, fill = "skyblue", color = "black") +
  labs(title = "Histogram of Spam Prediction Confidence", x = "Probability of Spam", y = "Frequency") +
  theme_minimal()
ggsave(file.path(output_dir, "prediction_confidence_histogram.png"), conf_plot, width = 7, height = 5, dpi = 300)

# ✅ Print Evaluation Summary
cat("\n📊 Classification Report:\n")
cat("Accuracy:", round(conf_matrix$overall['Accuracy'], 4), "\n")
cat("Recall (nonspam):", round(conf_matrix$byClass['Sensitivity'], 4), "\n")
cat("Recall (spam):", round(conf_matrix$byClass['Specificity'], 4), "\n")
cat("Precision (spam):", round(conf_matrix$byClass['Pos Pred Value'], 4), "\n")
cat("F1 Score:", round(2 * conf_matrix$byClass['Sensitivity'] * conf_matrix$byClass['Pos Pred Value'] /
                        (conf_matrix$byClass['Sensitivity'] + conf_matrix$byClass['Pos Pred Value']), 4), "\n")
cat("AUC:", round(auc_value, 4), "\n")
cat("✅ All plots saved to:", output_dir, "\n")
