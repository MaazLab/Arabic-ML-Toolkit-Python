# Load required libraries
# Install packages if not already installed
required_packages <- c("tidyverse", "caret", "cluster", "dbscan", "class", 
                      "factoextra", "plotly", "viridis", "isotree")
install.packages("remotes")

install.packages("isotree")
library(isotree)

for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, repos = "http://cran.us.r-project.org")
    library(pkg, character.only = TRUE)
  }
}

# Load the iris dataset (built into R)
data(iris)

# Explore the dataset
head(iris)
str(iris)
summary(iris)

# Check for missing values
sum(is.na(iris))

# Preprocess the data
# 1. Extract numeric features only (removing the Species column)
iris_numeric <- iris %>% select_if(is.numeric)

# 2. Scale the data
iris_scaled <- scale(iris_numeric)

# 3. Convert to data frame for easier manipulation
iris_df <- as.data.frame(iris_scaled)

# Split data into training and test sets (80/20 split)
set.seed(123)  # For reproducibility
train_indices <- createDataPartition(1:nrow(iris_df), p = 0.8, list = FALSE)
train_data <- iris_df[train_indices, ]
test_data <- iris_df[-train_indices, ]

# ------------------------------------------------------------------------
# K-Means Clustering
# ------------------------------------------------------------------------

# Find optimal number of clusters using elbow method
set.seed(123)
wss <- sapply(1:10, function(k) {
  kmeans(train_data, centers = k, nstart = 25)$tot.withinss
})

# Plot the elbow curve
elbow_data <- data.frame(k = 1:10, wss = wss)
elbow_plot <- ggplot(elbow_data, aes(x = k, y = wss)) +
  geom_line() +
  geom_point() +
  labs(x = "Number of Clusters", y = "Within-Cluster Sum of Squares",
       title = "Elbow Method for Optimal k") +
  theme_minimal()
print(elbow_plot)

# Alternative method: silhouette method
sil <- sapply(2:10, function(k) {
  km <- kmeans(train_data, centers = k, nstart = 25)
  ss <- silhouette(km$cluster, dist(train_data))
  mean(ss[, 3])
})

# Plot silhouette scores
silhouette_data <- data.frame(k = 2:10, sil_score = sil)
silhouette_plot <- ggplot(silhouette_data, aes(x = k, y = sil_score)) +
  geom_line() +
  geom_point() +
  labs(x = "Number of Clusters", y = "Average Silhouette Score",
       title = "Silhouette Method for Optimal k") +
  theme_minimal()
print(silhouette_plot)

# For iris dataset, we know there are 3 species, but let's confirm with our methods
# Find the k with highest silhouette score
optimal_k <- which.max(sil) + 1  # Add 1 because we started from k=2
cat("Optimal k based on silhouette score:", optimal_k, "\n")

# Apply K-means clustering with the optimal k
set.seed(123)
kmeans_model <- kmeans(train_data, centers = optimal_k, nstart = 25)

# Add cluster assignments to the training data
train_data$kmeans_cluster <- as.factor(kmeans_model$cluster)

# Visualize clusters using PCA for dimensionality reduction
pca_result <- prcomp(train_data[, -ncol(train_data)], scale. = FALSE)
pca_data <- as.data.frame(pca_result$x[, 1:2])
pca_data$cluster <- train_data$kmeans_cluster

# Plot PCA-reduced clusters
kmeans_plot <- ggplot(pca_data, aes(x = PC1, y = PC2, color = cluster)) +
  geom_point(alpha = 0.7) +
  labs(title = "K-means Clustering Visualized with PCA",
       x = "Principal Component 1", y = "Principal Component 2") +
  theme_minimal() +
  scale_color_viridis_d()
print(kmeans_plot)

# Apply the k-means model to test data
test_data$kmeans_cluster <- as.factor(
  apply(test_data, 1, function(x) {
    distances <- apply(kmeans_model$centers, 1, function(center) {
      sqrt(sum((x - center)^2))
    })
    return(which.min(distances))
  })
)

# Evaluate clustering quality using silhouette score on test data
test_dist <- dist(test_data[, -ncol(test_data)])
sil_score_test <- silhouette(as.numeric(test_data$kmeans_cluster), test_dist)
cat("Silhouette score on test data:", mean(sil_score_test[, 3]), "\n")

# Compare with actual species if you want to assess accuracy
# Add species information to training data
train_data$actual_species <- iris$Species[train_indices]

# Create a confusion matrix
confusion_table <- table(Predicted = train_data$kmeans_cluster, 
                        Actual = train_data$actual_species)
print("Confusion Matrix for K-means Clustering:")
print(confusion_table)

# Calculate accuracy - note that cluster numbers might not match species numbers
# so we need to find the best mapping
cluster_to_species_map <- apply(confusion_table, 1, which.max)
correct_predictions <- sum(sapply(1:nrow(train_data), function(i) {
  actual_idx <- as.numeric(train_data$actual_species[i])
  cluster_idx <- as.numeric(train_data$kmeans_cluster[i])
  return(cluster_to_species_map[cluster_idx] == actual_idx)
}))

accuracy <- correct_predictions / nrow(train_data)
cat("K-means clustering accuracy:", accuracy, "\n")

# ------------------------------------------------------------------------
# DBSCAN Clustering
# ------------------------------------------------------------------------
# -------------------------------
# DBSCAN Clustering
# -------------------------------

# Define numeric feature columns (if not already done)
feature_cols <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")

# Compute k-NN distances (for k = 4)
knn_dist <- kNNdist(as.matrix(train_data[, feature_cols]), k = 4)

# Sort distances for elbow plot
knn_dist_sorted <- sort(knn_dist)

# Plot the sorted distances
knn_plot <- ggplot(data.frame(index = 1:length(knn_dist_sorted), 
                              distance = knn_dist_sorted), 
                   aes(x = index, y = distance)) +
  geom_line() +
  geom_point() +
  labs(title = "4-NN Distances (Sorted)",
       x = "Points (sorted by distance)",
       y = "Distance to 4th Nearest Neighbor") +
  theme_minimal()
print(knn_plot)

# Find "elbow" point by analyzing slope changes
slopes <- diff(knn_dist_sorted)
slope_change <- diff(slopes)
elbow_candidates <- which(abs(slope_change) > quantile(abs(slope_change), 0.9))

if (length(elbow_candidates) > 0) {
  eps_index <- min(elbow_candidates)
  eps_value <- knn_dist_sorted[eps_index]
} else {
  eps_value <- median(knn_dist_sorted)
}

cat("Suggested eps value:", eps_value, "\n")

# Choose minPts (usually: number of features + 1)
min_pts <- length(feature_cols) + 1

# Run DBSCAN clustering
dbscan_model <- dbscan(as.matrix(train_data[, feature_cols]), eps = eps_value, minPts = min_pts)

# Save cluster assignments
train_data$dbscan_cluster <- as.factor(dbscan_model$cluster)

# Perform PCA for visualization
pca_result <- prcomp(train_data[, feature_cols], scale. = FALSE)
pca_data <- as.data.frame(pca_result$x[, 1:2])
colnames(pca_data) <- c("PC1", "PC2")

# Add DBSCAN cluster info for visualization
pca_data$dbscan_cluster <- train_data$dbscan_cluster

# Plot DBSCAN clusters in PCA space
dbscan_plot <- ggplot(pca_data, aes(x = PC1, y = PC2, color = dbscan_cluster)) +
  geom_point(alpha = 0.7) +
  labs(title = "DBSCAN Clustering Visualized with PCA",
       x = "Principal Component 1",
       y = "Principal Component 2") +
  theme_minimal() +
  scale_color_viridis_d(name = "Cluster", labels = function(x) ifelse(x == "0", "Noise", x))
print(dbscan_plot)


# Plot PCA-reduced DBSCAN clusters
dbscan_plot <- ggplot(pca_data, aes(x = PC1, y = PC2, color = dbscan_cluster)) +
  geom_point(alpha = 0.7) +
  labs(title = "DBSCAN Clustering Visualized with PCA",
       x = "Principal Component 1", y = "Principal Component 2") +
  theme_minimal() +
  scale_color_viridis_d(name = "Cluster", 
                      labels = function(x) ifelse(x == "0", "Noise", x))
print(dbscan_plot)

# Apply DBSCAN to test data (based on training model)
# For DBSCAN, we need to use the same eps and minPts on the test data
dbscan_test <- dbscan(test_data[, -ncol(test_data)], eps = eps_value, minPts = min_pts)
test_data$dbscan_cluster <- as.factor(dbscan_test$cluster)

# Create a confusion matrix for DBSCAN (excluding noise points)
# Add species information to training data if not already added
if (!("actual_species" %in% colnames(train_data))) {
  train_data$actual_species <- iris$Species[train_indices]
}

# Only consider non-noise points
non_noise <- train_data$dbscan_cluster != "0"
if (sum(non_noise) > 0) {
  dbscan_confusion <- table(Predicted = train_data$dbscan_cluster[non_noise], 
                          Actual = train_data$actual_species[non_noise])
  print("Confusion Matrix for DBSCAN Clustering (excluding noise):")
  print(dbscan_confusion)
}

# Count noise points
noise_count <- sum(train_data$dbscan_cluster == "0")
cat("Number of noise points identified by DBSCAN:", noise_count, "\n")

# ------------------------------------------------------------------------
# K-Nearest Neighbors (KNN) for Anomaly Detection
# ------------------------------------------------------------------------

# Function to compute average distance to k nearest neighbors
avg_knn_dist <- function(data, k) {
  dist_matrix <- as.matrix(dist(data))
  diag(dist_matrix) <- Inf  # Exclude self-distances
  
  # Find k nearest neighbors for each point
  sorted_dist <- t(apply(dist_matrix, 1, sort))
  avg_dist <- rowMeans(sorted_dist[, 1:k, drop = FALSE])
  
  return(avg_dist)
}

# Calculate average distance to k nearest neighbors for training data
k_value <- 5  # Number of neighbors to consider
train_data$knn_dist <- avg_knn_dist(train_data[, -c(ncol(train_data), ncol(train_data)-1)], k_value)

# Define outliers as points with distances greater than a threshold
# Here we use the 95th percentile as threshold
threshold <- quantile(train_data$knn_dist, 0.95)
train_data$knn_outlier <- factor(train_data$knn_dist > threshold, 
                                labels = c("Normal", "Outlier"))

# Visualize KNN outliers
pca_data$knn_outlier <- train_data$knn_outlier
knn_plot <- ggplot(pca_data, aes(x = PC1, y = PC2, color = knn_outlier)) +
  geom_point(alpha = 0.7) +
  labs(title = "KNN Outlier Detection",
       x = "Principal Component 1", y = "Principal Component 2") +
  theme_minimal() +
  scale_color_manual(values = c("Normal" = "blue", "Outlier" = "red"))
print(knn_plot)

# Apply KNN outlier detection to test data
test_data$knn_dist <- avg_knn_dist(test_data[, -c(ncol(test_data), ncol(test_data)-1)], k_value)
test_data$knn_outlier <- factor(test_data$knn_dist > threshold, 
                              labels = c("Normal", "Outlier"))

# ------------------------------------------------------------------------
# Isolation Forest for Anomaly Detection
# ------------------------------------------------------------------------
# Apply Isolation Forest for anomaly detection (on training data)
# Ensure feature columns are defined (just once)
feature_cols <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")

# -------------------------------------------------------------------
# Isolation Forest: TRAINING
# -------------------------------------------------------------------

# Convert feature columns to numeric matrix for training
train_numeric <- train_data[, feature_cols]
train_numeric[] <- lapply(train_numeric, as.numeric)  # ensures all columns are numeric

# Fit Isolation Forest
iso_model <- isolation.forest(
  as.matrix(train_numeric), 
  ndim = 2, 
  ntrees = 100, 
  nthreads = 1  # avoid OpenMP issues on macOS
)

# Predict anomaly scores
anomaly_scores <- predict(iso_model, as.matrix(train_numeric))
train_data$anomaly_score <- anomaly_scores

# Set threshold (95th percentile)
iso_threshold <- quantile(anomaly_scores, 0.95)

# Assign anomaly labels
train_data$anomaly <- factor(anomaly_scores > iso_threshold, labels = c("Normal", "Anomaly"))

# -------------------------------------------------------------------
# Visualization using PCA
# -------------------------------------------------------------------

# Add anomaly info to existing PCA data
pca_data$anomaly <- train_data$anomaly

# Plot Isolation Forest anomalies
iso_plot <- ggplot(pca_data, aes(x = PC1, y = PC2, color = anomaly)) +
  geom_point(alpha = 0.7) +
  labs(title = "Isolation Forest Anomaly Detection",
       x = "Principal Component 1", y = "Principal Component 2") +
  theme_minimal() +
  scale_color_manual(values = c("Normal" = "blue", "Anomaly" = "red"))
print(iso_plot)

# -------------------------------------------------------------------
# Isolation Forest: TEST DATA
# -------------------------------------------------------------------

# Ensure test feature columns are numeric
test_numeric <- test_data[, feature_cols]
test_numeric[] <- lapply(test_numeric, as.numeric)

# Predict anomaly scores on test data
test_anomaly_scores <- predict(iso_model, as.matrix(test_numeric))
test_data$anomaly_score <- test_anomaly_scores

# Assign anomaly labels
test_data$anomaly <- factor(test_anomaly_scores > iso_threshold, labels = c("Normal", "Anomaly"))
# ------------------------------------------------------------------------
# Local Outlier Factor (LOF) for Anomaly Detection
# ------------------------------------------------------------------------

# Calculat# -------------------------------
# Local Outlier Factor (LOF)
# -------------------------------

# Step 1: Define feature columns (if not already)
feature_cols <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width")

# Step 2: Ensure all training features are numeric
train_numeric <- train_data[, feature_cols]
train_numeric[] <- lapply(train_numeric, as.numeric)

# Step 3: Apply LOF (use minPts instead of k)
lof_scores <- lof(as.matrix(train_numeric), minPts = 6)
train_data$lof_score <- lof_scores

# Step 4: Define LOF anomalies using 95th percentile
lof_threshold <- quantile(lof_scores, 0.95)
train_data$lof_anomaly <- factor(lof_scores > lof_threshold, labels = c("Normal", "Anomaly"))

# Step 5: Update PCA plot with LOF labels
pca_data$lof_anomaly <- train_data$lof_anomaly

lof_plot <- ggplot(pca_data, aes(x = PC1, y = PC2, color = lof_anomaly)) +
  geom_point(alpha = 0.7) +
  labs(title = "LOF Anomaly Detection",
       x = "Principal Component 1", y = "Principal Component 2") +
  theme_minimal() +
  scale_color_manual(values = c("Normal" = "blue", "Anomaly" = "red"))
print(lof_plot)

# -------------------------------
# Apply LOF to Test Data
# -------------------------------

# Ensure all test features are numeric
test_numeric <- test_data[, feature_cols]
test_numeric[] <- lapply(test_numeric, as.numeric)

# Predict LOF scores for test data
test_lof_scores <- lof(as.matrix(test_numeric), minPts = 6)
test_data$lof_score <- test_lof_scores

# Label test data as anomalies
test_data$lof_anomaly <- factor(test_lof_scores > lof_threshold, labels = c("Normal", "Anomaly"))


# ------------------------------------------------------------------------
# Compare and Evaluate All Methods
# ------------------------------------------------------------------------

# Prepare a comprehensive results dataframe
results_df <- data.frame(
  observation = 1:nrow(train_data),
  actual_species = train_data$actual_species,
  kmeans_cluster = train_data$kmeans_cluster,
  dbscan_cluster = train_data$dbscan_cluster,
  knn_outlier = train_data$knn_outlier,
  isolation_forest = train_data$anomaly,
  lof_anomaly = train_data$lof_anomaly,
  PC1 = pca_data$PC1,
  PC2 = pca_data$PC2
)

# Create an interactive plot to compare different methods
plot_ly(results_df) %>%
  add_trace(x = ~PC1, y = ~PC2, color = ~actual_species, 
            type = "scatter", mode = "markers", 
            marker = list(size = 10, opacity = 0.6),
            name = "Actual Species", visible = TRUE) %>%
  add_trace(x = ~PC1, y = ~PC2, color = ~kmeans_cluster, 
            type = "scatter", mode = "markers", 
            marker = list(size = 10, opacity = 0.6),
            name = "K-means", visible = FALSE) %>%
  add_trace(x = ~PC1, y = ~PC2, color = ~dbscan_cluster, 
            type = "scatter", mode = "markers", 
            marker = list(size = 10, opacity = 0.6),
            name = "DBSCAN", visible = FALSE) %>%
  add_trace(x = ~PC1, y = ~PC2, color = ~knn_outlier, 
            type = "scatter", mode = "markers", 
            marker = list(size = 10, opacity = 0.6),
            name = "KNN Outliers", visible = FALSE) %>%
  add_trace(x = ~PC1, y = ~PC2, color = ~isolation_forest, 
            type = "scatter", mode = "markers", 
            marker = list(size = 10, opacity = 0.6),
            name = "Isolation Forest", visible = FALSE) %>%
  add_trace(x = ~PC1, y = ~PC2, color = ~lof_anomaly, 
            type = "scatter", mode = "markers", 
            marker = list(size = 10, opacity = 0.6),
            name = "LOF", visible = FALSE) %>%
  layout(
    title = "Comparison of Clustering and Anomaly Detection Methods",
    xaxis = list(title = "Principal Component 1"),
    yaxis = list(title = "Principal Component 2"),
    updatemenus = list(
      list(
        type = "buttons",
        direction = "right",
        x = 0.1,
        y = 1.1,
        buttons = list(
          list(method = "update", args = list(list(visible = c(TRUE, FALSE, FALSE, FALSE, FALSE, FALSE))), 
               label = "Actual Species"),
          list(method = "update", args = list(list(visible = c(FALSE, TRUE, FALSE, FALSE, FALSE, FALSE))), 
               label = "K-means"),
          list(method = "update", args = list(list(visible = c(FALSE, FALSE, TRUE, FALSE, FALSE, FALSE))), 
               label = "DBSCAN"),
          list(method = "update", args = list(list(visible = c(FALSE, FALSE, FALSE, TRUE, FALSE, FALSE))), 
               label = "KNN"),
          list(method = "update", args = list(list(visible = c(FALSE, FALSE, FALSE, FALSE, TRUE, FALSE))), 
               label = "Isolation Forest"),
          list(method = "update", args = list(list(visible = c(FALSE, FALSE, FALSE, FALSE, FALSE, TRUE))), 
               label = "LOF")
        )
      )
    )
  )

# Calculate evaluation metrics
# For clustering: silhouette scores
kmeans_sil <- silhouette(as.numeric(train_data$kmeans_cluster), 
                       dist(train_data[, -c(ncol(train_data):ncol(train_data)-5)]))
kmeans_avg_sil <- mean(kmeans_sil[, 3])

# For DBSCAN, ignore noise points (cluster 0) when calculating silhouette
dbscan_clusters <- as.numeric(train_data$dbscan_cluster)
non_noise <- dbscan_clusters != 0
if (sum(non_noise) > 1 && length(unique(dbscan_clusters[non_noise])) > 1) {
  dbscan_sil <- silhouette(dbscan_clusters[non_noise], 
                         dist(train_data[non_noise, -c(ncol(train_data):ncol(train_data)-5)]))
  dbscan_avg_sil <- mean(dbscan_sil[, 3])
} else {
  dbscan_avg_sil <- NA
}

# Count number of outliers/anomalies detected by each method
knn_outliers <- sum(train_data$knn_outlier == "Outlier")
iso_outliers <- sum(train_data$anomaly == "Anomaly")
lof_outliers <- sum(train_data$lof_anomaly == "Anomaly")
dbscan_noise <- sum(train_data$dbscan_cluster == 0)

# Print evaluation metrics
cat("\nEvaluation Metrics:\n")
cat("-------------------\n")
cat("K-means average silhouette score:", kmeans_avg_sil, "\n")
cat("DBSCAN average silhouette score:", dbscan_avg_sil, "\n")
cat("Number of KNN outliers:", knn_outliers, "\n")
cat("Number of Isolation Forest anomalies:", iso_outliers, "\n")
cat("Number of LOF anomalies:", lof_outliers, "\n")
cat("Number of DBSCAN noise points:", dbscan_noise, "\n")

# Check overlap between different anomaly detection methods
knn_anomalies <- which(train_data$knn_outlier == "Outlier")
iso_anomalies <- which(train_data$anomaly == "Anomaly")
lof_anomalies <- which(train_data$lof_anomaly == "Anomaly")
dbscan_noise_indices <- which(train_data$dbscan_cluster == 0)

# Calculate pairwise overlaps
overlap_knn_iso <- length(intersect(knn_anomalies, iso_anomalies))
overlap_knn_lof <- length(intersect(knn_anomalies, lof_anomalies))
overlap_knn_dbscan <- length(intersect(knn_anomalies, dbscan_noise_indices))
overlap_iso_lof <- length(intersect(iso_anomalies, lof_anomalies))
overlap_iso_dbscan <- length(intersect(iso_anomalies, dbscan_noise_indices))
overlap_lof_dbscan <- length(intersect(lof_anomalies, dbscan_noise_indices))

# Calculate overlap among all methods
overlap_all <- length(Reduce(intersect, list(
  knn_anomalies, 
  iso_anomalies, 
  lof_anomalies, 
  dbscan_noise_indices
)))

cat("\nOverlap between anomaly detection methods:\n")
cat("----------------------------------------\n")
cat("KNN and Isolation Forest:", overlap_knn_iso, "points\n")
cat("KNN and LOF:", overlap_knn_lof, "points\n")
cat("KNN and DBSCAN noise:", overlap_knn_dbscan, "points\n")
cat("Isolation Forest and LOF:", overlap_iso_lof, "points\n")
cat("Isolation Forest and DBSCAN noise:", overlap_iso_dbscan, "points\n")
cat("LOF and DBSCAN noise:", overlap_lof_dbscan, "points\n")
cat("All four methods:", overlap_all, "points\n")

# Summary and conclusions
cat("\nSummary and Conclusions:\n")
cat("----------------------\n")
cat("1. K-means identified", optimal_k, "clusters in the data\n")
cat("2. DBSCAN identified", length(unique(train_data$dbscan_cluster)) - (0 %in% unique(as.numeric(train_data$dbscan_cluster))), "clusters and", dbscan_noise, "noise points\n")
cat("3. KNN identified", knn_outliers, "outliers based on distance to neighbors\n")
cat("4. Isolation Forest identified", iso_outliers, "anomalies in the data\n")
cat("5. LOF identified", lof_outliers, "local outliers in the data\n")

# Compare clustering results with actual species classifications
cat("\nComparison with actual species:\n")
cat("-----------------------------\n")

# Function to calculate adjusted Rand index
ari <- function(clustering, reference) {
  n <- length(clustering)
  nc <- max(clustering)
  nr <- max(reference)
  
  # Create contingency table
  cont <- matrix(0, nc, nr)
  for (i in 1:n) {
    cont[clustering[i], reference[i]] <- cont[clustering[i], reference[i]] + 1
  }
  
  # Calculate row and column sums
  row_sums <- rowSums(cont)
  col_sums <- colSums(cont)
  
  # Calculate terms for ARI formula
  term1 <- sum(choose(cont, 2))
  term2 <- sum(choose(row_sums, 2)) * sum(choose(col_sums, 2)) / choose(n, 2)
  term3 <- (sum(choose(row_sums, 2)) + sum(choose(col_sums, 2))) / 2
  
  # Calculate ARI
  ari_value <- (term1 - term2) / (term3 - term2)
  return(ari_value)
}

# Calculate ARI for k-means and DBSCAN (for non-noise points)
kmeans_ari <- ari(as.numeric(train_data$kmeans_cluster), as.numeric(train_data$actual_species))
cat("K-means Adjusted Rand Index:", kmeans_ari, "\n")

if (sum(non_noise) > 0) {
  dbscan_ari <- ari(as.numeric(train_data$dbscan_cluster[non_noise]), 
                  as.numeric(train_data$actual_species[non_noise]))
  cat("DBSCAN Adjusted Rand Index (excluding noise):", dbscan_ari, "\n")
}

# Calculate the percentage of each species classified as anomalies
species_anomaly_counts <- table(
  Species = train_data$actual_species,
  KNN = train_data$knn_outlier,
  ISO = train_data$anomaly,
  LOF = train_data$lof_anomaly
)

print("Anomaly detection by species:")
print(species_anomaly_counts)

cat("\nPercentage of each species detected as anomalies:\n")
for (method in c("KNN", "ISO", "LOF")) {
  cat("\n", method, "method:\n", sep = "")
  for (species in levels(train_data$actual_species)) {
    species_count <- sum(train_data$actual_species == species)
    anomaly_count <- sum(train_data$actual_species == species & 
                        (if (method == "KNN") train_data$knn_outlier == "Outlier" 
                         else if (method == "ISO") train_data$anomaly == "Anomaly"
                         else train_data$lof_anomaly == "Anomaly"))
    percent <- (anomaly_count / species_count) * 100
    cat("  ", species, ": ", sprintf("%.2f", percent), "%\n", sep = "")
  }
}

# Create function to plot confusion matrices
plot_confusion_matrix <- function(conf_matrix, title) {
  conf_matrix_df <- as.data.frame(as.table(conf_matrix))
  names(conf_matrix_df) <- c("Predicted", "Actual", "Freq")
  
  ggplot(conf_matrix_df, aes(x = Actual, y = Predicted, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), color = "white", size = 4) +
    scale_fill_viridis_c() +
    labs(title = title, x = "Actual", y = "Predicted", fill = "Count") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# Plot confusion matrices
if (exists("confusion_table")) {
  conf_plot <- plot_confusion_matrix(confusion_table, "K-means Confusion Matrix")
  print(conf_plot)
}

if (exists("dbscan_confusion")) {
  dbscan_conf_plot <- plot_confusion_matrix(dbscan_confusion, "DBSCAN Confusion Matrix")
  print(dbscan_conf_plot)
}

# Save all plots to pdf
# Create folder for saving plots
output_dir <- "Clustering_Anomaly_Detection_Plots"
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}
# Save each plot as PNG
ggsave(filename = file.path(output_dir, "elbow_plot.png"), plot = elbow_plot, width = 7, height = 5)
ggsave(filename = file.path(output_dir, "silhouette_plot.png"), plot = silhouette_plot, width = 7, height = 5)
ggsave(filename = file.path(output_dir, "kmeans_plot.png"), plot = kmeans_plot, width = 7, height = 5)
ggsave(filename = file.path(output_dir, "dbscan_plot.png"), plot = dbscan_plot, width = 7, height = 5)
ggsave(filename = file.path(output_dir, "knn_plot.png"), plot = knn_plot, width = 7, height = 5)
ggsave(filename = file.path(output_dir, "iso_plot.png"), plot = iso_plot, width = 7, height = 5)
ggsave(filename = file.path(output_dir, "lof_plot.png"), plot = lof_plot, width = 7, height = 5)

# Optional: Confusion matrix plots
if (exists("conf_plot")) {
  ggsave(filename = file.path(output_dir, "confusion_matrix_kmeans.png"), plot = conf_plot, width = 7, height = 5)
}
if (exists("dbscan_conf_plot")) {
  ggsave(filename = file.path(output_dir, "confusion_matrix_dbscan.png"), plot = dbscan_conf_plot, width = 7, height = 5)
}
