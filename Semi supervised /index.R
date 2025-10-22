# Semi-Supervised Learning Methods Comparison
# ===========================================
# Comparing Self-Training, Co-Training, and Graph-Based Label Propagation
# on two standard datasets

# Install required packages if not already installed
if (!require("caret")) install.packages("caret")
if (!require("e1071")) install.packages("e1071")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("randomForest")) install.packages("randomForest")
if (!require("igraph")) install.packages("igraph")
if (!require("kernlab")) install.packages("kernlab")
if (!require("mlbench")) install.packages("mlbench")
if (!require("MASS")) install.packages("MASS")
if (!require("dplyr")) install.packages("dplyr")
if (!require("tidyr")) install.packages("tidyr")
if (!require("pROC")) install.packages("pROC")

# Load necessary libraries
library(caret)
library(e1071)
library(ggplot2)
library(randomForest)
library(igraph)
library(kernlab)
library(mlbench)
library(MASS)
library(dplyr)
library(tidyr)
library(pROC)

# Set seed for reproducibility
set.seed(123)

# ======================================================
# 1. Dataset Preparation Functions
# ======================================================

# Function to load and preprocess datasets
load_dataset <- function(dataset_name) {
  if (dataset_name == "pima") {
    data <- iris
    names(data) <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species")
    data$Species <- as.factor(as.character(data$Species))
  } else if (dataset_name == "pima") {
    data("PimaIndiansDiabetes", package = "mlbench")
    data <- PimaIndiansDiabetes
    names(data)[ncol(data)] <- "Class"
    data$Class <- as.factor(data$Class)
  } else {
    stop("Dataset not supported. Use 'iris' or 'pima'.")
  }
  return(data)
}


# Function to split data into labeled, unlabeled, and test sets
split_data <- function(data, label_percent = 0.1, test_percent = 0.2, stratify = TRUE) {
  if (stratify) {
    # Stratified sampling to maintain class distribution
    n_classes <- length(unique(data[[ncol(data)]]))
    class_col <- ncol(data)
    class_name <- names(data)[class_col]
    
    labeled_indices <- c()
    test_indices <- c()
    
    for (class_val in unique(data[[class_col]])) {
      class_indices <- which(data[[class_col]] == class_val)
      n_class <- length(class_indices)
      
      # Calculate sizes for this class
      labeled_size <- floor(n_class * label_percent)
      test_size <- floor(n_class * test_percent)
      
      # Random sampling within this class
      sampled_indices <- sample(class_indices)
      labeled_indices <- c(labeled_indices, sampled_indices[1:labeled_size])
      test_indices <- c(test_indices, sampled_indices[(labeled_size + 1):(labeled_size + test_size)])
    }
    
    # Remaining indices are for unlabeled data
    all_indices <- 1:nrow(data)
    unlabeled_indices <- setdiff(all_indices, c(labeled_indices, test_indices))
  } else {
    # Simple random sampling
    n <- nrow(data)
    labeled_size <- floor(n * label_percent) 
    test_size <- floor(n * test_percent)
    
    indices <- sample(1:n)
    labeled_indices <- indices[1:labeled_size]
    test_indices <- indices[(labeled_size + 1):(labeled_size + test_size)]
    unlabeled_indices <- indices[(labeled_size + test_size + 1):n]
  }
  
  # Create datasets
  labeled_data <- data[labeled_indices, ]
  unlabeled_data <- data[unlabeled_indices, ]
  test_data <- data[test_indices, ]
  
  # Create a version of unlabeled data with labels removed
  unlabeled_data_no_labels <- unlabeled_data
  unlabeled_data_no_labels[[ncol(unlabeled_data_no_labels)]] <- NA
  
  return(list(
    labeled = labeled_data,
    unlabeled = unlabeled_data,  # With true labels (only for evaluation)
    unlabeled_no_labels = unlabeled_data_no_labels,  # Actual unlabeled data for learning
    test = test_data,
    true_labels = unlabeled_data[[ncol(unlabeled_data)]]  # True labels of unlabeled data (for evaluation)
  ))
}

# Feature splitting for co-training (based on correlation or random)
split_features <- function(data, method = "correlation", n_views = 2) {
  # Get feature names (excluding the class column)
  feature_cols <- setdiff(names(data), names(data)[ncol(data)])
  n_features <- length(feature_cols)
  
  if (method == "random") {
    # Random split
    shuffled_features <- sample(feature_cols)
    view_size <- ceiling(n_features / n_views)
    
    views <- list()
    for (i in 1:n_views) {
      start_idx <- (i-1) * view_size + 1
      end_idx <- min(i * view_size, n_features)
      if (start_idx <= n_features) {
        views[[i]] <- shuffled_features[start_idx:end_idx]
      }
    }
    return(views)
  } 
  else if (method == "correlation") {
    # Correlation-based split (features with high correlation go to different views)
    numeric_data <- data[, sapply(data, is.numeric)]
    cor_matrix <- cor(numeric_data, method = "pearson")
    
    # Simple greedy approach: alternate assignment of features to views
    feature_order <- order(rowSums(abs(cor_matrix)), decreasing = TRUE)
    ordered_features <- feature_cols[feature_order]
    
    views <- vector("list", n_views)
    for (i in 1:length(ordered_features)) {
      view_idx <- (i-1) %% n_views + 1
      views[[view_idx]] <- c(views[[view_idx]], ordered_features[i])
    }
    return(views)
  }
  else {
    stop("Method not supported. Use 'random' or 'correlation'.")
  }
}

# ======================================================
# 2. Semi-Supervised Learning Algorithms
# ======================================================

# 2.1 Self-Training Implementation
self_training <- function(labeled_data, unlabeled_data, test_data, 
                          max_iterations = 10, 
                          k_per_iteration = 5,
                          confidence_threshold = 0.8) {
  
  # Get feature and class column names
  feature_cols <- setdiff(names(labeled_data), names(labeled_data)[ncol(labeled_data)])
  class_col <- names(labeled_data)[ncol(labeled_data)]
  
  # Initialize performance tracking
  accuracy_history <- numeric(max_iterations + 1)
  labeled_size_history <- numeric(max_iterations + 1)
  
  # Initial labeled data
  current_labeled <- labeled_data
  current_unlabeled <- unlabeled_data
  
  # Initial classifier (Random Forest)
  model <- randomForest(
    formula = as.formula(paste(class_col, "~ .")),
    data = current_labeled,
    ntree = 100
  )
  
  # Evaluate initial performance
  initial_preds <- predict(model, test_data[, feature_cols, drop = FALSE])
  initial_acc <- sum(initial_preds == test_data[[class_col]]) / nrow(test_data)
  
  accuracy_history[1] <- initial_acc
  labeled_size_history[1] <- nrow(current_labeled)
  
  cat("Self-Training: Initial accuracy =", round(initial_acc, 4), 
      "with", nrow(current_labeled), "labeled examples\n")
  
  # Store assigned labels and confidence for final evaluation
  all_assigned_labels <- c()
  all_true_labels <- c()
  
  # Self-training iterations
  for (iter in 1:max_iterations) {
    if (nrow(current_unlabeled) == 0) {
      cat("No more unlabeled data. Stopping at iteration", iter, "\n")
      break
    }
    
    # Get predictions on unlabeled data
    preds_probs <- predict(model, current_unlabeled[, feature_cols, drop = FALSE], type = "prob")
    
    # Calculate confidence scores
    confidence <- apply(preds_probs, 1, max)
    predictions <- predict(model, current_unlabeled[, feature_cols, drop = FALSE])
    
    # Find examples above confidence threshold
    confident_indices <- which(confidence >= confidence_threshold)
    
    # If we have too many confident examples, take the k most confident
    if (length(confident_indices) > k_per_iteration) {
      confident_indices <- confident_indices[order(confidence[confident_indices], decreasing = TRUE)[1:k_per_iteration]]
    }
    
    # If we have confident predictions
    if (length(confident_indices) > 0) {
      # Save assigned and true labels for evaluation
      assigned_labels <- predictions[confident_indices]
      true_labels <- current_unlabeled[[class_col]][confident_indices]
      
      all_assigned_labels <- c(all_assigned_labels, as.character(assigned_labels))
      all_true_labels <- c(all_true_labels, as.character(true_labels))
      
      # Create data to add to labeled set
      to_add <- current_unlabeled[confident_indices, ]
      to_add[[class_col]] <- predictions[confident_indices]
      
      # Add to labeled and remove from unlabeled
      current_labeled <- rbind(current_labeled, to_add)
      current_unlabeled <- current_unlabeled[-confident_indices, ]
      
      # Retrain model
      model <- randomForest(
        formula = as.formula(paste(class_col, "~ .")),
        data = current_labeled,
        ntree = 100
      )
      
      # Evaluate on test data
      preds <- predict(model, test_data[, feature_cols, drop = FALSE])
      acc <- sum(preds == test_data[[class_col]]) / nrow(test_data)
      
      # Store metrics
      accuracy_history[iter + 1] <- acc
      labeled_size_history[iter + 1] <- nrow(current_labeled)
      
      cat("Self-Training: Iteration", iter, "- Added", length(confident_indices), 
          "examples. Accuracy =", round(acc, 4), "\n")
    } else {
      cat("No confident examples found in iteration", iter, 
          ". Lowering threshold to", confidence_threshold * 0.9, "\n")
      confidence_threshold <- confidence_threshold * 0.9
      
      # If threshold gets too low, stop
      if (confidence_threshold < 0.5) {
        cat("Confidence threshold too low. Stopping.\n")
        break
      }
      
      # Don't advance iteration counter
      iter <- iter - 1
    }
  }
  
  # Calculate pseudo-label accuracy
  pseudo_label_accuracy <- NA
  if (length(all_assigned_labels) > 0) {
    pseudo_label_accuracy <- sum(all_assigned_labels == all_true_labels) / length(all_assigned_labels)
  }
  
  # Return results
  return(list(
    final_model = model,
    accuracy_history = accuracy_history[1:(iter+1)],
    labeled_size_history = labeled_size_history[1:(iter+1)],
    pseudo_label_accuracy = pseudo_label_accuracy,
    final_labeled_data = current_labeled
  ))
}

# 2.2 Co-Training Implementation
co_training <- function(labeled_data, unlabeled_data, test_data, 
                       view1_cols, view2_cols,
                       max_iterations = 10, 
                       k_per_iteration = 5,
                       confidence_threshold = 0.8) {
  
  # Get class column name
  class_col <- names(labeled_data)[ncol(labeled_data)]
  
  # Initialize performance tracking
  accuracy_history <- numeric(max_iterations + 1)
  labeled_size_history <- numeric(max_iterations + 1)
  
  # Initial labeled data
  current_labeled <- labeled_data
  current_unlabeled <- unlabeled_data
  
  # Formula for each view
  formula_view1 <- as.formula(paste(class_col, "~", paste(view1_cols, collapse = " + ")))
  formula_view2 <- as.formula(paste(class_col, "~", paste(view2_cols, collapse = " + ")))
  
  # Initial classifiers (Random Forest) for each view
  model_view1 <- randomForest(
    formula = formula_view1,
    data = current_labeled,
    ntree = 100
  )
  
  model_view2 <- randomForest(
    formula = formula_view2,
    data = current_labeled,
    ntree = 100
  )
  
  # Evaluate initial performance (using ensemble of both classifiers)
  preds_view1 <- predict(model_view1, test_data[, view1_cols, drop = FALSE], type = "prob")
  preds_view2 <- predict(model_view2, test_data[, view2_cols, drop = FALSE], type = "prob")
  
  # Simple averaging of probabilities
  ensemble_probs <- (preds_view1 + preds_view2) / 2
  ensemble_preds <- colnames(ensemble_probs)[apply(ensemble_probs, 1, which.max)]
  
  initial_acc <- sum(ensemble_preds == test_data[[class_col]]) / nrow(test_data)
  
  accuracy_history[1] <- initial_acc
  labeled_size_history[1] <- nrow(current_labeled)
  
  cat("Co-Training: Initial accuracy =", round(initial_acc, 4), 
      "with", nrow(current_labeled), "labeled examples\n")
  
  # Store assigned labels and confidence for final evaluation
  all_assigned_labels <- c()
  all_true_labels <- c()
  
  # Co-training iterations
  for (iter in 1:max_iterations) {
    if (nrow(current_unlabeled) == 0) {
      cat("No more unlabeled data. Stopping at iteration", iter, "\n")
      break
    }
    
    # Get predictions from each view on unlabeled data
    preds_view1 <- predict(model_view1, current_unlabeled[, view1_cols, drop = FALSE], type = "prob")
    preds_view2 <- predict(model_view2, current_unlabeled[, view2_cols, drop = FALSE], type = "prob")
    
    # Calculate confidence scores for each view
    confidence_view1 <- apply(preds_view1, 1, max)
    confidence_view2 <- apply(preds_view2, 1, max)
    
    # Predicted labels for each view
    preds_labels_view1 <- colnames(preds_view1)[apply(preds_view1, 1, which.max)]
    preds_labels_view2 <- colnames(preds_view2)[apply(preds_view2, 1, which.max)]
    
    # Find examples above confidence threshold for each view
    confident_indices_view1 <- which(confidence_view1 >= confidence_threshold)
    confident_indices_view2 <- which(confidence_view2 >= confidence_threshold)
    
    # Take the k most confident from each view
    k_per_view <- ceiling(k_per_iteration / 2)
    
    if (length(confident_indices_view1) > k_per_view) {
      confident_indices_view1 <- confident_indices_view1[order(confidence_view1[confident_indices_view1], 
                                                            decreasing = TRUE)[1:k_per_view]]
    }
    
    if (length(confident_indices_view2) > k_per_view) {
      confident_indices_view2 <- confident_indices_view2[order(confidence_view2[confident_indices_view2], 
                                                            decreasing = TRUE)[1:k_per_view]]
    }
    
    # Combine indices, ensuring no duplicates
    all_confident_indices <- unique(c(confident_indices_view1, confident_indices_view2))
    
    # If we have confident predictions
    if (length(all_confident_indices) > 0) {
      # Determine which view's prediction to use for each example
      # If example is confident in both views, use the more confident one
      labels_to_assign <- character(length(all_confident_indices))
      
      for (i in 1:length(all_confident_indices)) {
        idx <- all_confident_indices[i]
        if (idx %in% confident_indices_view1 && idx %in% confident_indices_view2) {
          # Both views are confident - use the more confident one
          if (confidence_view1[idx] >= confidence_view2[idx]) {
            labels_to_assign[i] <- preds_labels_view1[idx]
          } else {
            labels_to_assign[i] <- preds_labels_view2[idx]
          }
        } else if (idx %in% confident_indices_view1) {
          labels_to_assign[i] <- preds_labels_view1[idx]
        } else {
          labels_to_assign[i] <- preds_labels_view2[idx]
        }
      }
      
      # Save assigned and true labels for evaluation
      true_labels <- current_unlabeled[[class_col]][all_confident_indices]
      
      all_assigned_labels <- c(all_assigned_labels, labels_to_assign)
      all_true_labels <- c(all_true_labels, as.character(true_labels))
      
      # Create data to add to labeled set
      to_add <- current_unlabeled[all_confident_indices, ]
      to_add[[class_col]] <- factor(labels_to_assign, levels = levels(labeled_data[[class_col]]))
      
      # Add to labeled and remove from unlabeled
      current_labeled <- rbind(current_labeled, to_add)
      current_unlabeled <- current_unlabeled[-all_confident_indices, ]
      
      # Retrain models
      model_view1 <- randomForest(
        formula = formula_view1,
        data = current_labeled,
        ntree = 100
      )
      
      model_view2 <- randomForest(
        formula = formula_view2,
        data = current_labeled,
        ntree = 100
      )
      
      # Evaluate on test data
      preds_view1 <- predict(model_view1, test_data[, view1_cols, drop = FALSE], type = "prob")
      preds_view2 <- predict(model_view2, test_data[, view2_cols, drop = FALSE], type = "prob")
      
      # Simple averaging of probabilities
      ensemble_probs <- (preds_view1 + preds_view2) / 2
      ensemble_preds <- colnames(ensemble_probs)[apply(ensemble_probs, 1, which.max)]
      
      acc <- sum(ensemble_preds == test_data[[class_col]]) / nrow(test_data)
      
      # Store metrics
      accuracy_history[iter + 1] <- acc
      labeled_size_history[iter + 1] <- nrow(current_labeled)
      
      cat("Co-Training: Iteration", iter, "- Added", length(all_confident_indices), 
          "examples. Accuracy =", round(acc, 4), "\n")
    } else {
      cat("No confident examples found in iteration", iter, 
          ". Lowering threshold to", confidence_threshold * 0.9, "\n")
      confidence_threshold <- confidence_threshold * 0.9
      
      # If threshold gets too low, stop
      if (confidence_threshold < 0.5) {
        cat("Confidence threshold too low. Stopping.\n")
        break
      }
      
      # Don't advance iteration counter
      iter <- iter - 1
    }
  }
  
  # Calculate pseudo-label accuracy
  pseudo_label_accuracy <- NA
  if (length(all_assigned_labels) > 0) {
    pseudo_label_accuracy <- sum(all_assigned_labels == all_true_labels) / length(all_assigned_labels)
  }
  
  # Return results
  return(list(
    final_model_view1 = model_view1,
    final_model_view2 = model_view2,
    accuracy_history = accuracy_history[1:(iter+1)],
    labeled_size_history = labeled_size_history[1:(iter+1)],
    pseudo_label_accuracy = pseudo_label_accuracy,
    final_labeled_data = current_labeled,
    view1_cols = view1_cols,
    view2_cols = view2_cols
  ))
}

# 2.3 Graph-Based Label Propagation
label_propagation <- function(labeled_data, unlabeled_data, test_data,
                             sigma = 1.0, alpha = 0.99, max_iter = 30) {
  
  # Get feature and class column names
  feature_cols <- setdiff(names(labeled_data), names(labeled_data)[ncol(labeled_data)])
  class_col <- names(labeled_data)[ncol(labeled_data)]
  
  # Combine labeled and unlabeled data for graph construction
  combined_data <- rbind(
    labeled_data[, feature_cols, drop = FALSE],
    unlabeled_data[, feature_cols, drop = FALSE]
  )
  
  n_labeled <- nrow(labeled_data)
  n_unlabeled <- nrow(unlabeled_data)
  n_total <- n_labeled + n_unlabeled
  
  # Normalize data for better similarity computation
  combined_data_scaled <- scale(combined_data)
  
  # Construct the similarity graph using Gaussian kernel
  # This computes similarity between all pairs of points
  similarity_matrix <- matrix(0, n_total, n_total)
  
  for (i in 1:n_total) {
    for (j in i:n_total) {
      # Euclidean distance between points
      dist_ij <- sqrt(sum((combined_data_scaled[i,] - combined_data_scaled[j,])^2))
      # Gaussian kernel for similarity
      sim_ij <- exp(-dist_ij^2 / (2 * sigma^2))
      similarity_matrix[i, j] <- sim_ij
      similarity_matrix[j, i] <- sim_ij  # Symmetric matrix
    }
  }
  
  # Create one-hot encoding for labels
  classes <- levels(labeled_data[[class_col]])
  n_classes <- length(classes)
  
  # Initialize label matrix Y
  Y <- matrix(0, n_total, n_classes)
  colnames(Y) <- classes
  
  # Assign known labels
  for (i in 1:n_labeled) {
    label <- as.character(labeled_data[[class_col]][i])
    Y[i, label] <- 1
  }
  
  # Store original labeled data
  Y_0 <- Y
  
  # Label propagation algorithm
  cat("Label Propagation: Starting with", n_labeled, "labeled examples\n")
  
  prev_Y <- Y
  for (iter in 1:max_iter) {
    # Propagate labels: Y = alpha * S * Y + (1 - alpha) * Y_0
    Y <- alpha * similarity_matrix %*% Y + (1 - alpha) * Y_0
    
    # Ensure labeled data keep their original labels
    Y[1:n_labeled, ] <- Y_0[1:n_labeled, ]
    
    # Check convergence
    diff <- sum(abs(Y - prev_Y))
    if (diff < 1e-6) {
      cat("Label Propagation: Converged after", iter, "iterations\n")
      break
    }
    
    prev_Y <- Y
  }
  
  # Assign labels to unlabeled data based on maximum propagated value
  propagated_labels <- apply(Y[(n_labeled+1):n_total, ], 1, which.max)
  propagated_labels <- classes[propagated_labels]
  
  # Calculate accuracy of propagated labels (for evaluation only)
  true_labels <- as.character(unlabeled_data[[class_col]])
  pseudo_label_accuracy <- sum(propagated_labels == true_labels) / length(propagated_labels)
  
  cat("Label Propagation: Pseudo-label accuracy =", round(pseudo_label_accuracy, 4), "\n")
  
  # Create combined dataset with propagated labels
  unlabeled_with_labels <- unlabeled_data
  unlabeled_with_labels[[class_col]] <- factor(propagated_labels, levels = classes)
  final_labeled_data <- rbind(labeled_data, unlabeled_with_labels)
  
  # Train a final classifier on the propagated labels
  final_model <- randomForest(
    formula = as.formula(paste(class_col, "~ .")),
    data = final_labeled_data,
    ntree = 100
  )
  
  # Evaluate on test data
  test_preds <- predict(final_model, test_data[, feature_cols, drop = FALSE])
  test_acc <- sum(test_preds == test_data[[class_col]]) / nrow(test_data)
  
  cat("Label Propagation: Final test accuracy =", round(test_acc, 4), "\n")
  
  # Return results
  return(list(
    final_model = final_model,
    final_accuracy = test_acc,
    propagated_labels = propagated_labels,
    true_labels = true_labels,
    pseudo_label_accuracy = pseudo_label_accuracy,
    final_labeled_data = final_labeled_data
  ))
}

# ======================================================
# 3. Evaluation and Visualization Functions
# ======================================================

# Visualization function for 2D projections of data
visualize_dataset <- function(data, projection_method = "pca") {
  # Get feature and class column names
  feature_cols <- setdiff(names(data), names(data)[ncol(data)])
  class_col <- names(data)[ncol(data)]
  
  # Extract features
  features <- data[, feature_cols]
  labels <- data[[class_col]]
  
  if (projection_method == "pca") {
    # Principal Component Analysis for dimensionality reduction
    pca_result <- prcomp(features, center = TRUE, scale. = TRUE)
    proj_data <- data.frame(
      PC1 = pca_result$x[, 1],
      PC2 = pca_result$x[, 2],
      Class = labels
    )
    
    # Create plot
    p <- ggplot(proj_data, aes(x = PC1, y = PC2, color = Class)) +
      geom_point(alpha = 0.7, size = 3) +
      labs(
        title = paste("PCA projection of", deparse(substitute(data))),
        x = "Principal Component 1",
        y = "Principal Component 2"
      ) +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5))
  }
  else if (projection_method == "tsne") {
    # t-SNE dimensionality reduction
    if (!require("Rtsne")) install.packages("Rtsne")
    library(Rtsne)
    
    # Perform t-SNE
    tsne_result <- Rtsne(as.matrix(features), dims = 2, perplexity = min(30, floor(nrow(features)/4)),
                        check_duplicates = FALSE)
    
    proj_data <- data.frame(
      X = tsne_result$Y[, 1],
      Y = tsne_result$Y[, 2],
      Class = labels
    )
    
    # Create plot
    p <- ggplot(proj_data, aes(x = X, y = Y, color = Class)) +
      geom_point(alpha = 0.7, size = 3) +
      labs(
        title = paste("t-SNE projection of", deparse(substitute(data))),
        x = "t-SNE Dimension 1",
        y = "t-SNE Dimension 2"
      ) +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5))
  }
  
  return(p)
}

# Function to compare performance of different methods
plot_comparison <- function(self_results, co_results, label_prop_result, supervised_acc, dataset_name) {
  # Prepare data for self-training and co-training
  iterations_self <- 0:(length(self_results$accuracy_history) - 1)
  iterations_co <- 0:(length(co_results$accuracy_history) - 1)
  
  comparison_df <- data.frame(
    Iteration = c(iterations_self, iterations_co),
    Method = c(rep("Self-Training", length(iterations_self)), rep("Co-Training", length(iterations_co))),
    Accuracy = c(self_results$accuracy_history, co_results$accuracy_history)
  )
  
  # Add label propagation as a horizontal line
  lp_accuracy <- label_prop_result$final_accuracy
  
  # Create plot
  p <- ggplot(comparison_df, aes(x = Iteration, y = Accuracy, color = Method, group = Method)) +
    geom_line(size = 1) +
    geom_point(size = 3) +
    geom_hline(yintercept = supervised_acc, linetype = "dashed", color = "black") +
    geom_hline(yintercept = lp_accuracy, linetype = "dashed", color = "green") +
    annotate("text", x = max(comparison_df$Iteration), y = supervised_acc + 0.02, 
            label = "Supervised baseline", hjust = 1) +
    annotate("text", x = max(comparison_df$Iteration), y = lp_accuracy + 0.02, 
            label = "Label Propagation", hjust = 1, color = "green") +
    labs(
      title = paste("Semi-Supervised Learning Methods Comparison on", dataset_name),
      x = "Iteration",
      y = "Accuracy",
      color = "Method"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  return(p)
}

# Function to visualize pseudo-label quality
plot_pseudo_labels <- function(self_results, co_results, label_prop_result, dataset_name) {
  # Create data frame for plotting
  methods <- c("Self-Training", "Co-Training", "Label Propagation")
  accuracies <- c(
    self_results$pseudo_label_accuracy,
    co_results$pseudo_label_accuracy,
    label_prop_result$pseudo_label_accuracy
  )
  
# Continuing from previous code...
  # Continuing from previous code...
  
  pseudo_df <- data.frame(
    Method = methods,
    Accuracy = accuracies
  )
  
  # Create bar plot
  p <- ggplot(pseudo_df, aes(x = Method, y = Accuracy, fill = Method)) +
    geom_bar(stat = "identity", width = 0.6) +
    geom_text(aes(label = sprintf("%.1f%%", Accuracy * 100)), 
              position = position_stack(vjust = 0.5), color = "white", size = 4) +
    labs(
      title = paste("Pseudo-Label Accuracy Comparison on", dataset_name),
      x = "",
      y = "Accuracy of Assigned Labels",
      fill = "Method"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  return(p)
}

# Function to visualize the labeled vs unlabeled data
plot_labeled_unlabeled <- function(labeled_data, unlabeled_data, test_data, dataset_name, projection_method = "pca") {
  # Get feature and class column names
  feature_cols <- setdiff(names(labeled_data), names(labeled_data)[ncol(labeled_data)])
  class_col <- names(labeled_data)[ncol(labeled_data)]
  
  # Combine for visualization
  labeled_data$DataType <- "Labeled"
  unlabeled_data$DataType <- "Unlabeled"
  test_data$DataType <- "Test"
  
  all_data <- rbind(labeled_data, unlabeled_data, test_data)
  
  # Extract features
  features <- all_data[, feature_cols]
  
  if (projection_method == "pca") {
    # Principal Component Analysis
    pca_result <- prcomp(features, center = TRUE, scale. = TRUE)
    proj_data <- data.frame(
      PC1 = pca_result$x[, 1],
      PC2 = pca_result$x[, 2],
      Class = all_data[[class_col]],
      DataType = all_data$DataType
    )
    
    # Create plot
    p <- ggplot(proj_data, aes(x = PC1, y = PC2, color = Class, shape = DataType)) +
      geom_point(alpha = 0.7, size = 3) +
      labs(
        title = paste("Data Split for", dataset_name, "Dataset"),
        x = "Principal Component 1",
        y = "Principal Component 2"
      ) +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5))
  }
  else if (projection_method == "tsne") {
    # t-SNE dimensionality reduction
    if (!require("Rtsne")) install.packages("Rtsne")
    library(Rtsne)
    
    # Perform t-SNE
    tsne_result <- Rtsne(as.matrix(features), dims = 2, perplexity = min(30, floor(nrow(features)/4)),
                         check_duplicates = FALSE)
    
    proj_data <- data.frame(
      X = tsne_result$Y[, 1],
      Y = tsne_result$Y[, 2],
      Class = all_data[[class_col]],
      DataType = all_data$DataType
    )
    
    # Create plot
    p <- ggplot(proj_data, aes(x = X, y = Y, color = Class, shape = DataType)) +
      geom_point(alpha = 0.7, size = 3) +
      labs(
        title = paste("Data Split for", dataset_name, "Dataset"),
        x = "t-SNE Dimension 1",
        y = "t-SNE Dimension 2"
      ) +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5))
  }
  
  return(p)
}

# Function to create confusion matrices
plot_confusion_matrices <- function(self_results, co_results, label_prop_result, test_data, dataset_name) {
  # Get class column name
  class_col <- names(test_data)[ncol(test_data)]
  
  # Get feature columns
  feature_cols <- setdiff(names(test_data), class_col)
  
  # Make predictions with each model
  self_preds <- predict(self_results$final_model, test_data[, feature_cols, drop = FALSE])
  
  # For co-training, use ensemble of both models
  co_preds_view1 <- predict(co_results$final_model_view1, test_data[, co_results$view1_cols, drop = FALSE], type = "prob")
  co_preds_view2 <- predict(co_results$final_model_view2, test_data[, co_results$view2_cols, drop = FALSE], type = "prob")
  ensemble_probs <- (co_preds_view1 + co_preds_view2) / 2
  co_preds <- colnames(ensemble_probs)[apply(ensemble_probs, 1, which.max)]
  
  # Label propagation predictions
  lp_preds <- predict(label_prop_result$final_model, test_data[, feature_cols, drop = FALSE])
  
  # True labels
  true_labels <- test_data[[class_col]]
  
  # Create confusion matrices
  self_cm <- confusionMatrix(self_preds, true_labels)
  co_cm <- confusionMatrix(factor(co_preds, levels = levels(true_labels)), true_labels)
  lp_cm <- confusionMatrix(lp_preds, true_labels)
  
  # Extract confusion matrix values
  extract_cm_data <- function(cm) {
    cm_table <- as.data.frame(cm$table)
    names(cm_table) <- c("Reference", "Prediction", "Freq")
    return(cm_table)
  }
  
  self_cm_data <- extract_cm_data(self_cm)
  co_cm_data <- extract_cm_data(co_cm)
  lp_cm_data <- extract_cm_data(lp_cm)
  
  # Add method information
  self_cm_data$Method <- "Self-Training"
  co_cm_data$Method <- "Co-Training"
  lp_cm_data$Method <- "Label Propagation"
  
  # Combine all confusion matrices
  all_cm_data <- rbind(self_cm_data, co_cm_data, lp_cm_data)
  
  # Plot combined confusion matrices
  p <- ggplot(all_cm_data, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), color = "white") +
    scale_fill_gradient(low = "darkblue", high = "red") +
    facet_wrap(~ Method, ncol = 3) +
    labs(
      title = paste("Confusion Matrices Comparison on", dataset_name),
      x = "Reference",
      y = "Prediction",
      fill = "Count"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  return(p)
}

# Function to run supervised baseline (for comparison)
run_supervised_baseline <- function(labeled_data, test_data) {
  # Get feature and class column names
  feature_cols <- setdiff(names(labeled_data), names(labeled_data)[ncol(labeled_data)])
  class_col <- names(labeled_data)[ncol(labeled_data)]
  
  # Train supervised model on only the labeled data
  model <- randomForest(
    formula = as.formula(paste(class_col, "~ .")),
    data = labeled_data,
    ntree = 100
  )
  
  # Evaluate
  preds <- predict(model, test_data[, feature_cols, drop = FALSE])
  accuracy <- sum(preds == test_data[[class_col]]) / nrow(test_data)
  
  cat("Supervised Baseline Accuracy =", round(accuracy, 4), 
      "with", nrow(labeled_data), "labeled examples\n")
  
  return(list(
    model = model,
    accuracy = accuracy
  ))
}

# ======================================================
# 4. Run experiments on both datasets
# ======================================================

# Function to run all methods on a dataset
run_experiments <- function(dataset_name, label_percent = 0.1, test_percent = 0.2) {
  cat("\n\n========================================================\n")
  cat("RUNNING EXPERIMENTS ON", toupper(dataset_name), "DATASET\n")
  cat("========================================================\n\n")
  
  # Load dataset
  data <- load_dataset(dataset_name)
  
  # Split data
  data_split <- split_data(data, label_percent = label_percent, test_percent = test_percent)
  
  # Run supervised baseline
  supervised_baseline <- run_supervised_baseline(data_split$labeled, data_split$test)
  
  # Run Self-Training
  cat("\n--- Running Self-Training ---\n")
  self_results <- self_training(
    data_split$labeled,
    data_split$unlabeled_no_labels,
    data_split$test,
    max_iterations = 20,
    k_per_iteration = max(3, round(nrow(data_split$unlabeled) * 0.05)),
    confidence_threshold = 0.8
  )
  
  # Split features for Co-Training
  feature_cols <- setdiff(names(data), names(data)[ncol(data)])
  feature_views <- split_features(data, method = "correlation", n_views = 2)
  view1_cols <- feature_views[[1]]
  view2_cols <- feature_views[[2]]
  
  cat("\nFeature split for Co-Training:\n")
  cat("View 1:", paste(view1_cols, collapse = ", "), "\n")
  cat("View 2:", paste(view2_cols, collapse = ", "), "\n\n")
  
  # Run Co-Training
  cat("--- Running Co-Training ---\n")
  co_results <- co_training(
    data_split$labeled,
    data_split$unlabeled_no_labels,
    data_split$test,
    view1_cols = view1_cols,
    view2_cols = view2_cols,
    max_iterations = 20,
    k_per_iteration = max(3, round(nrow(data_split$unlabeled) * 0.05)),
    confidence_threshold = 0.8
  )
  
  # Run Label Propagation
  cat("\n--- Running Label Propagation ---\n")
  label_prop_result <- label_propagation(
    data_split$labeled,
    data_split$unlabeled_no_labels,
    data_split$test,
    sigma = 1.0,
    alpha = 0.99
  )
  
  # Create visualizations
  cat("\n--- Creating Visualizations ---\n")
  
  # 1. Visualization of dataset
  dataset_viz <- visualize_dataset(data, projection_method = "pca")
  
  # 2. Visualization of data split (labeled vs unlabeled)
  data_split_viz <- plot_labeled_unlabeled(
    data_split$labeled, 
    data_split$unlabeled, 
    data_split$test, 
    dataset_name
  )
  
  # 3. Performance comparison
  perf_comparison <- plot_comparison(
    self_results, 
    co_results, 
    label_prop_result, 
    supervised_baseline$accuracy,
    dataset_name
  )
  
  # 4. Pseudo-label quality
  pseudo_label_viz <- plot_pseudo_labels(
    self_results, 
    co_results, 
    label_prop_result,
    dataset_name
  )
  
  # 5. Confusion matrices
  confusion_matrices <- plot_confusion_matrices(
    self_results, 
    co_results, 
    label_prop_result,
    data_split$test,
    dataset_name
  )
  
  # Summarize results
  cat("\n--- Results Summary for", dataset_name, "---\n")
  cat("Dataset size:", nrow(data), "examples with", ncol(data)-1, "features and", length(levels(data[[ncol(data)]])), "classes\n")
  cat("Labeled data:", nrow(data_split$labeled), "examples (", label_percent*100, "%)\n")
  cat("Unlabeled data:", nrow(data_split$unlabeled), "examples\n")
  cat("Test data:", nrow(data_split$test), "examples\n\n")
  
  cat("Supervised baseline accuracy:", round(supervised_baseline$accuracy, 4), "\n")
  cat("Self-Training final accuracy:", round(self_results$accuracy_history[length(self_results$accuracy_history)], 4), "\n")
  cat("Co-Training final accuracy:", round(co_results$accuracy_history[length(co_results$accuracy_history)], 4), "\n")
  cat("Label Propagation accuracy:", round(label_prop_result$final_accuracy, 4), "\n\n")
  
  cat("Pseudo-label accuracy:\n")
  cat("Self-Training:", round(self_results$pseudo_label_accuracy, 4), "\n")
  cat("Co-Training:", round(co_results$pseudo_label_accuracy, 4), "\n")
  cat("Label Propagation:", round(label_prop_result$pseudo_label_accuracy, 4), "\n")
  
  # Return all results and visualizations
  return(list(
    dataset_name = dataset_name,
    supervised_baseline = supervised_baseline,
    self_training = self_results,
    co_training = co_results,
    label_propagation = label_prop_result,
    visualizations = list(
      dataset = dataset_viz,
      data_split = data_split_viz,
      performance = perf_comparison,
      pseudo_labels = pseudo_label_viz,
      confusion_matrices = confusion_matrices
    )
  ))
}

# ======================================================
# 5. Run experiments
# ======================================================

# Set a global seed for reproducibility
set.seed(42)

# Define datasets to use
datasets <- c("iris", "pima")

# Store results for each dataset
all_results <- list()

# Run experiments on each dataset
for (dataset_name in datasets) {
  all_results[[dataset_name]] <- run_experiments(
    dataset_name, 
    label_percent = 0.1,   # 10% labeled data
    test_percent = 0.2     # 20% test data
  )
  
  # Display visualizations
  print(all_results[[dataset_name]]$visualizations$dataset)
  print(all_results[[dataset_name]]$visualizations$data_split)
  print(all_results[[dataset_name]]$visualizations$performance)
  print(all_results[[dataset_name]]$visualizations$pseudo_labels)
  print(all_results[[dataset_name]]$visualizations$confusion_matrices)
}

# ======================================================
# 6. Final comparison across datasets
# ======================================================

# Create a comprehensive final performance comparison
create_final_comparison <- function(all_results) {
  # Extract accuracy values for each method and dataset
  datasets <- names(all_results)
  methods <- c("Supervised Baseline", "Self-Training", "Co-Training", "Label Propagation")
  
  # Initialize data frame
  final_comparison <- data.frame(
    Dataset = character(),
    Method = character(),
    Accuracy = numeric(),
    PseudoLabelAccuracy = numeric()
  )
  
  # Fill in data
  for (dataset in datasets) {
    result <- all_results[[dataset]]
    
    # Supervised baseline
    final_comparison <- rbind(final_comparison, data.frame(
      Dataset = dataset,
      Method = "Supervised Baseline",
      Accuracy = result$supervised_baseline$accuracy,
      PseudoLabelAccuracy = NA
    ))
    
    # Self-training
    final_comparison <- rbind(final_comparison, data.frame(
      Dataset = dataset,
      Method = "Self-Training",
      Accuracy = result$self_training$accuracy_history[length(result$self_training$accuracy_history)],
      PseudoLabelAccuracy = result$self_training$pseudo_label_accuracy
    ))
    
    # Co-training
    final_comparison <- rbind(final_comparison, data.frame(
      Dataset = dataset,
      Method = "Co-Training",
      Accuracy = result$co_training$accuracy_history[length(result$co_training$accuracy_history)],
      PseudoLabelAccuracy = result$co_training$pseudo_label_accuracy
    ))
    
    # Label propagation
    final_comparison <- rbind(final_comparison, data.frame(
      Dataset = dataset,
      Method = "Label Propagation",
      Accuracy = result$label_propagation$final_accuracy,
      PseudoLabelAccuracy = result$label_propagation$pseudo_label_accuracy
    ))
  }
  
  # Create plot of final accuracy comparison
  accuracy_plot <- ggplot(final_comparison, aes(x = Method, y = Accuracy, fill = Dataset)) +
    geom_bar(stat = "identity", position = position_dodge()) +
    geom_text(aes(label = sprintf("%.1f%%", Accuracy * 100)), 
              position = position_dodge(width = 0.9), vjust = -0.5) +
    labs(
      title = "Comparison of Semi-Supervised Learning Methods Across Datasets",
      subtitle = "Test Accuracy (%)",
      x = "",
      y = "Accuracy"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5))
  
  # Create plot of pseudo-label accuracy comparison
  pseudo_label_df <- final_comparison[!is.na(final_comparison$PseudoLabelAccuracy), ]
  
  pseudo_label_plot <- ggplot(pseudo_label_df, aes(x = Method, y = PseudoLabelAccuracy, fill = Dataset)) +
    geom_bar(stat = "identity", position = position_dodge()) +
    geom_text(aes(label = sprintf("%.1f%%", PseudoLabelAccuracy * 100)), 
              position = position_dodge(width = 0.9), vjust = -0.5) +
    labs(
      title = "Quality of Pseudo-Labels Across Methods and Datasets",
      subtitle = "Pseudo-Label Accuracy (%)",
      x = "",
      y = "Accuracy"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5))
  
  return(list(
    data = final_comparison,
    accuracy_plot = accuracy_plot,
    pseudo_label_plot = pseudo_label_plot
  ))
}

# Create and display final comparison
final_comparison <- create_final_comparison(all_results)
print(final_comparison$accuracy_plot)
print(final_comparison$pseudo_label_plot)

# Print summary table
cat("\n\n========================================================\n")
cat("FINAL RESULTS SUMMARY\n")
cat("========================================================\n\n")

# Format Accuracy
summary_table <- final_comparison$data
summary_table$Accuracy <- sprintf("%.2f%%", summary_table$Accuracy * 100)

# Drop the PseudoLabelAccuracy column
summary_table$PseudoLabelAccuracy <- NULL

# Print the cleaned summary
print(summary_table)




# Create output directory if it doesn't exist
output_dir <- "plots"
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
  cat("✅ Created output directory:", output_dir, "\n")
}

# Convert Accuracy from character to numeric
plot_data <- summary_table
plot_data$Accuracy <- as.numeric(sub("%", "", plot_data$Accuracy))

# Plot 1: Bar plot
p1 <- ggplot(plot_data, aes(x = Method, y = Accuracy, fill = Dataset)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8)) +
  geom_text(aes(label = paste0(sprintf("%.2f", Accuracy), "%")),
            vjust = -0.5, position = position_dodge(0.8), size = 3.5) +
  labs(title = "Accuracy Comparison",
       x = "Method", y = "Accuracy (%)") +
  scale_fill_brewer(palette = "Set2") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 30, hjust = 1),
        plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))

ggsave(filename = file.path(output_dir, "accuracy_comparison.png"),
       plot = p1, width = 10, height = 6, dpi = 300)

# Plot 2: Dataset-wise Breakdown
p2 <- ggplot(plot_data, aes(x = Dataset, y = Accuracy, fill = Method)) +
  geom_bar(stat = "identity", position = position_dodge()) +
  labs(title = "Dataset-wise Accuracy Breakdown",
       x = "Dataset", y = "Accuracy (%)") +
  scale_fill_brewer(palette = "Dark2") +
  theme_minimal()

ggsave(filename = file.path(output_dir, "dataset_breakdown.png"),
       plot = p2, width = 10, height = 6, dpi = 300)

# Plot 3: Optional - Performance trend (if multiple runs per method in future)
# You could use this when you have iteration-level or run-level data

cat("✅ Plots saved successfully in:", output_dir, "\n")



# ======================================================
# DONE!
# ======================================================