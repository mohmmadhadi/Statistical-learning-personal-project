# Install packages
install.packages(c("dplyr", "ggplot2", "cluster", "factoextra"))

# Load required libraries
library(dplyr)
library(ggplot2)
library(cluster)
library(factoextra)


#Loading Data
unsupervised <- read.csv("F:/University/Projects/Data Science/Statistical Learning/Maven Project/unsupervised.csv")

###Data preparation

str(unsupervised)

# Convert 'became_member_on' to Date type
unsupervised$became_member_on <- as.Date(unsupervised$became_member_on)

# Encode 'gender' as numeric
unsupervised$gender <- as.numeric(factor(unsupervised$gender, levels = c("M", "F", "O")))

# Convert the features of the data: customer.data
customer.data <- as.matrix(unsupervised[4:14])

# Set the row names of customer.data
row.names(customer.data) <- unsupervised$customer_id

# Check column means and standard deviations
colMeans(customer.data)
apply(customer.data,2, sd)

# Execute PCA, with scaling: customer.pr
customer.pr <- prcomp(customer.data, scale = TRUE)

summary(customer.pr)

# Create a biplot of customer.pr
biplot(customer.pr)
