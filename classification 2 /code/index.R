# Complete R Classification Analysis
install.packages("rpart.plot")

# 1. Load necessary libraries
library(tidyverse)    # For data manipulation and visualization
library(caret)        # For classification models and preprocessing
library(class)        # For KNN
library(rpart)        # For decision trees
library(rpart.plot)   # For visualizing trees
library(MASS)         # For additional models

# 2. Load and explore a dataset (using iris as an example)
data(iris)
head(iris)
summary(iris)
str(iris)

# 3. Exploratory data visualization
# Pairwise scatter plots
ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) +
  geom_point() +
  labs(title = "Sepal Length vs Sepal Width")

ggplot(iris, aes(x = Petal.Length, y = Petal.Width, color = Species)) +
  geom_point() +
  labs(title = "Petal Length vs Petal Width")

# 4. Preprocessing
# Check for missing values
sum(is.na(iris))

# Scale numeric features
preproc <- preProcess(iris[, 1:4], method = c("center", "scale"))
iris_scaled <- predict(preproc, iris)

# 5. Split data into training and testing sets
set.seed(123)  # For reproducibility
train_index <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
train_data <- iris[train_index, ]
test_data <- iris[-train_index, ]

# 6. Model Training and Evaluation

# ----------------------------------------------
# K-Nearest Neighbors (KNN)
# ----------------------------------------------
knn_model <- function(train_data, test_data, k = 5) {
  # Prepare data
  train_x <- train_data[, 1:4]
  train_y <- train_data$Species
  test_x <- test_data[, 1:4]
  test_y <- test_data$Species
  
  # Train and predict
  knn_pred <- knn(train = train_x, test = test_x, cl = train_y, k = k)
  
  # Evaluate
  conf_matrix <- confusionMatrix(knn_pred, test_y)
  
  # Visualize some test cases (first 10 for demonstration)
  test_subset <- test_data[1:10, ]
  pred_subset <- knn_pred[1:10]
  
  # Return results
  return(list(
    predictions = knn_pred,
    confusion_matrix = conf_matrix,
    test_subset = test_subset,
    pred_subset = pred_subset
  ))
}

# Run KNN model
knn_results <- knn_model(train_data, test_data, k = 5)
print(knn_results$confusion_matrix)

# Visualize KNN classifications (first 10 test points)
knn_vis_data <- knn_results$test_subset
knn_vis_data$Predicted <- knn_results$pred_subset

# Plot first 10 test predictions
ggplot(knn_vis_data, aes(x = Petal.Length, y = Petal.Width)) +
  geom_point(aes(color = Species, shape = Predicted), size = 3) +
  labs(title = "KNN Classification Results",
       subtitle = "Actual (color) vs Predicted (shape)")

# ----------------------------------------------
# Linear Regression for Classification 
# ----------------------------------------------
linear_model <- function(train_data, test_data) {
  # Create dummy variables for Species
  train_data_dummy <- train_data
  train_data_dummy$is_setosa <- ifelse(train_data$Species == "setosa", 1, 0)
  train_data_dummy$is_versicolor <- ifelse(train_data$Species == "versicolor", 1, 0)
  train_data_dummy$is_virginica <- ifelse(train_data$Species == "virginica", 1, 0)
  
  # Train models for each species
  model_setosa <- lm(is_setosa ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, 
                    data = train_data_dummy)
  model_versicolor <- lm(is_versicolor ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, 
                        data = train_data_dummy)
  model_virginica <- lm(is_virginica ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, 
                       data = train_data_dummy)
  
  # Predict on test data
  test_data_pred <- test_data
  test_data_pred$pred_setosa <- predict(model_setosa, test_data)
  test_data_pred$pred_versicolor <- predict(model_versicolor, test_data)
  test_data_pred$pred_virginica <- predict(model_virginica, test_data)
  
  # Get the class with highest probability
  predicted_classes <- apply(test_data_pred[, c("pred_setosa", "pred_versicolor", "pred_virginica")], 
                            1, which.max)
  species_levels <- levels(test_data$Species)
  test_data_pred$predicted <- species_levels[predicted_classes]
  test_data_pred$predicted <- factor(test_data_pred$predicted, levels = species_levels)
  
  # Calculate accuracy
  accuracy <- sum(test_data_pred$predicted == test_data$Species) / nrow(test_data)
  conf_matrix <- confusionMatrix(test_data_pred$predicted, test_data$Species)
  
  return(list(
    predictions = test_data_pred$predicted,
    confusion_matrix = conf_matrix,
    test_data_with_preds = test_data_pred
  ))
}

# Run Linear Regression model
linear_results <- linear_model(train_data, test_data)
print(linear_results$confusion_matrix)

# Visualize linear regression results
ggplot(linear_results$test_data_with_preds, aes(x = Petal.Length, y = Petal.Width)) +
  geom_point(aes(color = Species, shape = predicted), size = 3) +
  labs(title = "Linear Regression Classification Results",
       subtitle = "Actual (color) vs Predicted (shape)")

# ----------------------------------------------
# Decision Trees
# ----------------------------------------------
decision_tree_model <- function(train_data, test_data) {
  # Train decision tree
  dt_model <- rpart(Species ~ ., data = train_data, method = "class")
  
  # Visualize the tree
  rpart.plot(dt_model, extra = 104, box.palette = "RdBu", shadow.col = "gray")
  
  # Predict on test data
  dt_pred <- predict(dt_model, test_data, type = "class")
  
  # Evaluate
  conf_matrix <- confusionMatrix(dt_pred, test_data$Species)
  
  # Get variable importance
  var_importance <- dt_model$variable.importance
  
  return(list(
    model = dt_model,
    predictions = dt_pred,
    confusion_matrix = conf_matrix,
    variable_importance = var_importance
  ))
}

# Run Decision Tree model
dt_results <- decision_tree_model(train_data, test_data)
print(dt_results$confusion_matrix)

# Visualize variable importance
var_imp_df <- data.frame(
  Feature = names(dt_results$variable_importance),
  Importance = dt_results$variable_importance
)

ggplot(var_imp_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Decision Tree Variable Importance", x = "Feature")

# ----------------------------------------------
# Logistic Regression
# ----------------------------------------------
logistic_model <- function(train_data, test_data) {
  # Since logistic regression is binary, we'll use multinomial logistic regression for multiclass
  # We'll use the multinom function from the nnet package
  library(nnet)
  
  # Train the model
  log_model <- multinom(Species ~ ., data = train_data, trace = FALSE)
  
  # Predict on test data
  log_pred <- predict(log_model, test_data)
  
  # Evaluate
  conf_matrix <- confusionMatrix(log_pred, test_data$Species)
  
  # Get probabilities
  log_probs <- predict(log_model, test_data, type = "probs")
  
  return(list(
    model = log_model,
    predictions = log_pred,
    probabilities = log_probs,
    confusion_matrix = conf_matrix
  ))
}

# Run Logistic Regression model
log_results <- logistic_model(train_data, test_data)
print(log_results$confusion_matrix)

# Create test data with predictions for visualization
test_with_preds <- test_data
test_with_preds$predicted <- log_results$predictions

# Visualize logistic regression results
ggplot(test_with_preds, aes(x = Petal.Length, y = Petal.Width)) +
  geom_point(aes(color = Species, shape = predicted), size = 3) +
  labs(title = "Logistic Regression Classification Results",
       subtitle = "Actual (color) vs Predicted (shape)")

# ----------------------------------------------
# 7. Compare Models
# ----------------------------------------------
model_comparison <- data.frame(
  Model = c("KNN", "Linear Regression", "Decision Tree", "Logistic Regression"),
  Accuracy = c(
    knn_results$confusion_matrix$overall["Accuracy"],
    linear_results$confusion_matrix$overall["Accuracy"],
    dt_results$confusion_matrix$overall["Accuracy"],
    log_results$confusion_matrix$overall["Accuracy"]
  )
)

# Visualize model comparison
ggplot(model_comparison, aes(x = reorder(Model, Accuracy), y = Accuracy)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Model Comparison", x = "Model", y = "Accuracy") +
  ylim(0, 1)

# 8. Create decision boundaries visualization for KNN
create_decision_boundary <- function() {
  # Create a grid of points
  x_range <- seq(min(iris$Petal.Length) - 0.5, max(iris$Petal.Length) + 0.5, by = 0.1)
  y_range <- seq(min(iris$Petal.Width) - 0.5, max(iris$Petal.Width) + 0.5, by = 0.1)
  grid <- expand.grid(Petal.Length = x_range, Petal.Width = y_range)
  
  # Add dummy values for the other features (using means)
  grid$Sepal.Length <- mean(iris$Sepal.Length)
  grid$Sepal.Width <- mean(iris$Sepal.Width)
  
  # Predict classes for all grid points using KNN
  grid_pred <- knn(train = train_data[, 1:4], 
                  test = grid, 
                  cl = train_data$Species, 
                  k = 5)
  
  grid$Species <- grid_pred
  
  # Plot the decision boundary
  ggplot() +
    geom_tile(data = grid, aes(x = Petal.Length, y = Petal.Width, fill = Species), alpha = 0.3) +
    geom_point(data = iris, aes(x = Petal.Length, y = Petal.Width, color = Species), size = 2) +
    labs(title = "KNN Decision Boundaries (k=5)", 
         subtitle = "Using Petal Length and Width",
         fill = "Predicted Class",
         color = "Actual Class") +
    theme_minimal()
}
decision_boundary_plot <- create_decision_boundary()
print(decision_boundary_plot)


# . Visualization 
if (!dir.exists("classifications_2")) {
  dir.create("classifications_2")
}
sepal_plot <- ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) +
  geom_point() +
  labs(title = "Sepal Length vs Sepal Width")
ggsave("classifications_2/sepal_plot.png", sepal_plot, width = 6, height = 4)

petal_plot <- ggplot(iris, aes(x = Petal.Length, y = Petal.Width, color = Species)) +
  geom_point() +
  labs(title = "Petal Length vs Petal Width")
ggsave("classifications_2/petal_plot.png", petal_plot, width = 6, height = 4)

knn_class_plot <- ggplot(knn_vis_data, aes(x = Petal.Length, y = Petal.Width)) +
  geom_point(aes(color = Species, shape = Predicted), size = 3) +
  labs(title = "KNN Classification Results",
       subtitle = "Actual (color) vs Predicted (shape)")
ggsave("classifications_2/knn_classification.png", knn_class_plot, width = 6, height = 4)


linear_plot <- ggplot(linear_results$test_data_with_preds, aes(x = Petal.Length, y = Petal.Width)) +
  geom_point(aes(color = Species, shape = predicted), size = 3) +
  labs(title = "Linear Regression Classification Results",
       subtitle = "Actual (color) vs Predicted (shape)")
ggsave("classifications_2/linear_regression.png", linear_plot, width = 6, height = 4)


dt_var_plot <- ggplot(var_imp_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Decision Tree Variable Importance", x = "Feature")
ggsave("classifications_2/decision_tree_importance.png", dt_var_plot, width = 6, height = 4)

logistic_plot <- ggplot(test_with_preds, aes(x = Petal.Length, y = Petal.Width)) +
  geom_point(aes(color = Species, shape = predicted), size = 3) +
  labs(title = "Logistic Regression Classification Results",
       subtitle = "Actual (color) vs Predicted (shape)")
ggsave("classifications_2/logistic_regression.png", logistic_plot, width = 6, height = 4)

comparison_plot <- ggplot(model_comparison, aes(x = reorder(Model, Accuracy), y = Accuracy)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Model Comparison", x = "Model", y = "Accuracy") +
  ylim(0, 1)
ggsave("classifications_2/model_comparison.png", comparison_plot, width = 6, height = 4)

decision_boundary_plot <- create_decision_boundary()
ggsave("classifications_2/knn_decision_boundary.png", decision_boundary_plot, width = 6, height = 4)

