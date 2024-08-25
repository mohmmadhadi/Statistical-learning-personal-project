# Install packages
install.packages(c("dplyr", "ggplot2", "cluster", "factoextra"))
install.packages("ggcorrplot")


# Load required libraries
library(dplyr)
library(ggplot2)
library(cluster)
library(factoextra)
library(gridExtra)
library(ggcorrplot)


#Loading Data
unsupervised <- read.csv("F:/University/Projects/Data Science/Statistical Learning/Maven Project/unsupervised.csv")

###Data preparation

str(unsupervised)

# Convert 'became_member_on' to Date type
unsupervised$became_member_on <- as.Date(unsupervised$became_member_on)

# Encode 'gender' as numeric
unsupervised$gender <- as.numeric(factor(unsupervised$gender, levels = c("M", "F", "O")))

# Normalize the data (excluding non-numeric columns)
data_numeric <- unsupervised %>%
  select(where(is.numeric)) %>%
  scale()
# Convert to data frame
data_numeric <- as.data.frame(data_numeric)

# Creating a feature for Membership Duration (in days)
unsupervised$membership_duration <- as.numeric(Sys.Date() - unsupervised$became_member_on)

# Adding the new feature to the numeric data
data_numeric$membership_duration <- unsupervised$membership_duration

#Normalizing data
data_numeric <- scale(data_numeric)

###Exploratory Data Analysis
# Visualize the data to check for outliers
boxplot(data_numeric, las = 2)

# Function to check if a value is an outlier
is_outlier <- function(column) {
  Q1 <- quantile(column, 0.25)
  Q3 <- quantile(column, 0.75)
  IQR <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR
  upper_bound <- Q3 + 1.5 * IQR
  return(column < lower_bound | column > upper_bound)
}

# Initialize a logical vector to track rows with outliers
rows_with_outliers <- rep(FALSE, nrow(data_numeric))

# Loop through each column and mark rows containing outliers
for (i in seq_len(ncol(data_numeric))) {
  outlier_flags <- is_outlier(data_numeric[, i])
  rows_with_outliers <- rows_with_outliers | outlier_flags  # Update for any row with an outlier
}

# Remove rows with outliers
data_cleaned <- data_numeric[!rows_with_outliers, ]

data <- as.data.frame(data_cleaned)

# Print the result
cat("Original number of rows:", nrow(data.scaled), "\n")
cat("Number of rows after removing outliers:", nrow(data_cleaned), "\n")
print(data_cleaned)

# Visualize the data to check for outliers
boxplot(data_cleaned, las = 2)

# Correlation matrix

corr_matrix <- cor(data_cleaned)

ggcorrplot(corr_matrix, type = "lower", outline.color = "white", lab = TRUE,
           colors = c("darkred","#FFFFE0","darkblue")) +
  labs(title = "Correlation Heatmap")


# Check column means and standard deviations
colMeans(data_cleaned)
apply(data_cleaned,2, sd)

###PCA
# Execute PCA, with scaling: customer.pr
customer.pr <- prcomp(data_cleaned)

summary(customer.pr)

# Create a biplot of customer.pr
biplot(customer.pr)

# Calculate variability of each component
pr.var <- customer.pr$sdev ^2

# Variance explained by each principal component: pve
pve <- pr.var/sum(pr.var)

# Plot the bar plot for individual component variance
plot(pve, xlab = "Principal Component",
     ylab = "Proportion of Variance Explained", ylim = c(0, 1), type = "h",
     lwd = 10, col = "skyblue", xaxt = 'n')

# Add x-axis labels
axis(1, at = 1:length(pve), labels = 1:length(pve))

# Add cumulative variance line to the same plot
lines(cumsum(pve), type = "b", pch = 16, col = "red", lwd = 2)

# Add a legend
legend("right", legend = c("Individual Variance", "Cumulative Variance"),
       col = c("skyblue", "red"), lty = c(1, 1), lwd = 2, pch = c(NA, 16))

###HCust
# Calculate the (Euclidean) distances: data.dist
data.dist <- dist(data_cleaned)

# Create a hierarchical clustering model: customer.hclust
customer.hclust <- hclust(data.dist, method = 'complete')

# Plot with abbreviated labels
plot(customer.hclust, main = "Hierarchical Clustering Dendrogram")

# Add a horizontal cut line at the appropriate height
abline(h = 8.5, col = "red", lty = 2)

# Add rectangles to highlight the clusters
rect.hclust(customer.hclust, k = 6, border = 2:5)

###KMeans
##Elbow Method
# Compute K-Means clustering for a range of k
wss <- sapply(1:10, function(k) {
  kmeans(data_cleaned, centers = k, nstart = 25)$tot.withinss
})

# Plot elbow method
plot(1:10, wss, type = "b", xlab = "Number of clusters (k)", ylab = "Within-cluster sum of squares")

# Set number of clusters
k <- 6

# Apply K-Means clustering
set.seed(123) # For reproducibility
kmeans_result <- kmeans(data_cleaned, centers = k, nstart = 25)

# Add cluster assignment to the original data
data$cluster <- kmeans_result$cluster

# View clustering result
table(data$cluster)

data_pca <- as.data.frame(customer.pr$x)
# Add cluster assignment
data_pca$cluster <- as.factor(data$cluster)

summary(customer.pr)
# Plot the clusters
ggplot(data_pca, aes(x = PC1, y = PC2, color = cluster)) +
  geom_point() +
  labs(title = "Clustering Visualization with PCA")

