# Install the necessary packages
install.packages("caTools")
install.packages("xgboost")
install.packages("caret")
install.packages("Matrix")
install.packages("dplyr")
install.packages("vcd")
install.packages("ggplot2")
install.packages("gridExtra")



# Load necessary libraries
library(corrplot)
library(dplyr)
library(ggplot2)
library(vcd)  # For Cramér's V
library(caTools)
library(xgboost)
library(caret)
library(Matrix)
library(dplyr)
library(e1071)
library(ggplot2)
library(gridExtra)

#Loading Data
supervised <- read.csv("F:/University/Projects/Data Science/Statistical Learning/Maven Project/supervised.csv")

###Data preparation

str(supervised)

# Convert 'became_member_on' to Date type
supervised$became_member_on <- as.Date(supervised$became_member_on)

# Convert 'gender' into a factor first (good practice for categorical variables)
supervised$gender <- factor(supervised$gender, levels = c("M", "F", "O"))

# Use model.matrix to create one-hot encoded variables
# model.matrix automatically converts factors into dummy/one-hot encoded variables
one_hot_gender <- model.matrix(~ gender - 1, data = supervised)

# Append one-hot encoded columns back to the original dataset
supervised <- cbind(supervised, one_hot_gender)


# Creating a feature for Membership Duration (in days)
supervised$membership_duration <- as.numeric(Sys.Date() - supervised$became_member_on)

# Bining membership_duration into 3 categories: "short-term," "medium-term," and "long-term"
# We'll use quantiles to divide the data into roughly equal groups
supervised$membership_category <- cut(supervised$membership_duration,
                              breaks = quantile(supervised$membership_duration, probs = seq(0, 1, by = 1/3), na.rm = TRUE),
                              labels = c("short-term", "medium-term", "long-term"),
                              include.lowest = TRUE)

# Z-score Standardization (mean = 0, sd = 1)
supervised$membership_duration_zscore <- (supervised$membership_duration - mean(supervised$membership_duration, na.rm = TRUE)) /
  sd(supervised$membership_duration, na.rm = TRUE)

# Binning age into categories: "young adult," "middle-aged," "senior"
supervised$age_group <- cut(supervised$age,
                    breaks = c(-Inf, 25, 45, 65, Inf),
                    labels = c("young adult", "middle-aged", "senior", "elder"),
                    right = FALSE)

# Binning income into categories: "low," "medium," and "high"
supervised$income_group <- cut(supervised$income,
                       breaks = quantile(supervised$income, probs = seq(0, 1, by = 1/3), na.rm = TRUE),
                       labels = c("low", "medium", "high"),
                       include.lowest = TRUE)

# Ordinal encoding for 'age_group'
supervised$age_group <- factor(supervised$age_group, ordered = TRUE,
                       levels = c("young adult", "middle-aged", "senior", "elder"))

# Ordinal encoding for 'income_group'
supervised$income_group <- factor(supervised$income_group, ordered = TRUE,
                          levels = c("low", "medium", "high"))

# Create a target variable based on which offer type the customer completed the most
supervised$preferred_offer_type <- apply(supervised[, c("offer_type_bogo_completed",
                                                        "offer_type_discount_completed",
                                                        "num_completions_after_info")],
                                         1, function(x) {
                                           # Identify which offer type has the maximum completion
                                           offer_types <- c("bogo", "discount", "informational")
                                           max_index <- which.max(x)
                                           offer_types[max_index]
                                         })
################################


# Extract numeric features
numeric_features <- supervised[, sapply(supervised, is.numeric)]

# Calculate Pearson correlation for numeric features
cor_matrix_numeric <- cor(numeric_features, use = "complete.obs")
corrplot(cor_matrix_numeric, method = "color", type = "lower", tl.cex = 0.7)

# Calculate Cramér's V for categorical features
cramers_v_matrix <- function(df) {
  n <- ncol(df)
  result <- matrix(0, n, n)
  colnames(result) <- colnames(df)
  rownames(result) <- colnames(df)
  for (i in 1:n) {
    for (j in 1:n) {
      if (i != j) {
        result[i, j] <- assocstats(table(df[, i], df[, j]))$cramer
      }
    }
  }
  return(result)
}

# Apply to the factor columns
categorical_features <- supervised[, sapply(supervised, is.factor)]
cramer_matrix <- cramers_v_matrix(categorical_features)

# Visualize Cramér's V for categorical features
corrplot(cramer_matrix, method = "color", type = "lower", tl.cex = 0.7)


####################################33
# Drop useless columns
supervised <- supervised %>% select(-gender, -became_member_on, -age, -income,
                                    -offer_type_bogo_received, -offer_type_discount_received,
                                    -offer_type_informational, -offer_type_bogo_completed,
                                    -offer_type_discount_completed, -num_completions_after_info,
                                    -last_transaction_time, -num_offers_viewed, -num_offers_completed,
                                    -membership_duration)

str(supervised)

supervised[is.na(supervised)] <- 0


# Check the distribution of the target variable
table(supervised$preferred_offer_type)

# Convert preferred_offer_type to a factor
supervised$preferred_offer_type <- as.factor(supervised$preferred_offer_type)



###MODEL

#######################################################################
#XGBoost

# Remove the customer_id as it is not a feature for prediction
supervised <- supervised %>% select(-customer_id)

# Set seed for reproducibility
set.seed(123)

# Create a train/test split (70% train, 30% test)
train_index <- createDataPartition(supervised$preferred_offer_type, p = 0.7, list = FALSE)

# Split the data into training and test sets
train_set <- supervised[train_index, ]
test_set  <- supervised[-train_index, ]

# Check the dimensions to ensure the split worked correctly
dim(train_set)
dim(test_set)

# Convert the data frame into a matrix, which is required by XGBoost
x_train <- model.matrix(preferred_offer_type ~ . -1, data = train_set)
x_test  <- model.matrix(preferred_offer_type ~ . -1, data = test_set)

# The target variable needs to be converted to numeric (XGBoost needs labels as integers)
y_train <- as.numeric(train_set$preferred_offer_type) - 1 # XGBoost requires 0-based indexing
y_test  <- as.numeric(test_set$preferred_offer_type) - 1

# Convert the dataset into DMatrix objects (used by XGBoost)
dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest  <- xgb.DMatrix(data = x_test, label = y_test)

# Set the parameters for the XGBoost model
params <- list(
  objective = "multi:softmax",  # Multi-class classification
  eval_metric = "mlogloss",     # Evaluation metric (log loss)
  num_class = 3,                # Number of classes
  eta = 0.3,                    # Learning rate
  max_depth = 6,                # Maximum depth of trees
  subsample = 0.8,              # Row sampling rate
  colsample_bytree = 0.8        # Feature sampling rate
)

# Train the XGBoost model
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,                # Number of boosting rounds
  watchlist = list(train = dtrain, eval = dtest),
  early_stopping_rounds = 10,   # Stop early if no improvement after 10 rounds
  verbose = 1                   # Print output
)

# Make predictions on the test set
y_pred <- predict(xgb_model, newdata = dtest)

# Convert predictions to factor for comparison
y_pred <- as.factor(y_pred)
y_test <- as.factor(y_test)

# Calculate metrics for all classes
metrics_1 <- confusionMatrix(y_pred, y_test, mode = "everything")
print(metrics_1)

# Extract precision, recall, F1-score, and other metrics
precision <- metrics_1$byClass[, "Precision"]
recall <- metrics_1$byClass[, "Recall"]
f1_score <- metrics_1$byClass[, "F1"]

# Print the metrics
cat("Precision by Class: ", precision, "\n")
cat("Recall by Class: ", recall, "\n")
cat("F1-score by Class: ", f1_score, "\n")

# Print the overall metrics like accuracy, Kappa, etc.
cat("Overall Accuracy: ", metrics$overall['Accuracy'], "\n")
cat("Overall Kappa: ", metrics$overall['Kappa'], "\n")

# Get feature importance and plot
importance_matrix <- xgb.importance(model = xgb_model)
xgb.plot.importance(importance_matrix)


###Biased results!

# Get the class frequencies from the training set
class_freq <- table(y_train)

# Calculate the inverse of frequencies as class weights
class_weights <- max(class_freq) / class_freq

# Print the class weights for reference
print(class_weights)


# Adjust the parameters to include class weights
params <- list(
  objective = "multi:softmax",  # Multi-class classification
  eval_metric = "mlogloss",     # Evaluation metric (log loss)
  num_class = 3,                # Number of classes
  eta = 0.3,                    # Learning rate
  max_depth = 6,                # Maximum depth of trees
  subsample = 0.8,              # Row sampling rate
  colsample_bytree = 0.8        # Feature sampling rate
)

# Train the model with class weights applied
xgb_model_weighted <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train = dtrain, eval = dtest),
  early_stopping_rounds = 10,
  weight = class_weights[as.factor(y_train) + 1],  # Apply class weights
  verbose = 1
)


# Make predictions on the test set
y_pred_weighted  <- predict(xgb_model_weighted, newdata = dtest)

# Convert predictions to factor for comparison
y_pred_weighted <- as.factor(y_pred_weighted)
y_test <- as.factor(y_test)

# Calculate metrics for all classes
metrics_weighted_model <- confusionMatrix(y_pred_weighted, y_test, mode = "everything")
print(metrics_weighted_model)

# Extract precision, recall, F1-score, and other metrics
precision <- metrics_weighted_model$byClass[, "Precision"]
recall <- metrics_weighted_model$byClass[, "Recall"]
f1_score <- metrics_weighted_model$byClass[, "F1"]

# Print the metrics
cat("Precision by Class: ", precision, "\n")
cat("Recall by Class: ", recall, "\n")
cat("F1-score by Class: ", f1_score, "\n")

# Print the overall metrics like accuracy, Kappa, etc.
cat("Overall Accuracy: ", metrics_weighted_model$overall['Accuracy'], "\n")
cat("Overall Kappa: ", metrics_weighted_model$overall['Kappa'], "\n")


###still biased and not good

# Manually oversample the minority class (informational)
informational_class <- supervised[supervised$preferred_offer_type == "informational", ]
oversample_informational <- informational_class[sample(1:nrow(informational_class), size = 1000, replace = TRUE), ]

# Combine back with the original dataset
oversampled_data <- rbind(supervised, oversample_informational)


# Set seed for reproducibility
set.seed(123)

# Create a train/test split (70% train, 30% test)
train_index_oversample <- createDataPartition(oversampled_data$preferred_offer_type, p = 0.7, list = FALSE)

# Split the data into training and test sets
train_set <- oversampled_data[train_index_oversample, ]
test_set  <- oversampled_data[-train_index_oversample, ]

# Check the dimensions to ensure the split worked correctly
dim(train_set)
dim(test_set)

# Convert the data frame into a matrix, which is required by XGBoost
x_train <- model.matrix(preferred_offer_type ~ . -1, data = train_set)
x_test  <- model.matrix(preferred_offer_type ~ . -1, data = test_set)

# The target variable needs to be converted to numeric (XGBoost needs labels as integers)
y_train <- as.numeric(train_set$preferred_offer_type) - 1 # XGBoost requires 0-based indexing
y_test  <- as.numeric(test_set$preferred_offer_type) - 1

# Convert the dataset into DMatrix objects (used by XGBoost)
dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest  <- xgb.DMatrix(data = x_test, label = y_test)

# Set the parameters for the XGBoost model
params <- list(
  objective = "multi:softmax",  # Multi-class classification
  eval_metric = "mlogloss",     # Evaluation metric (log loss)
  num_class = 3,                # Number of classes
  eta = 0.3,                    # Learning rate
  max_depth = 6,                # Maximum depth of trees
  subsample = 0.8,              # Row sampling rate
  colsample_bytree = 0.8        # Feature sampling rate
)

# Train the XGBoost model
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,                # Number of boosting rounds
  watchlist = list(train = dtrain, eval = dtest),
  early_stopping_rounds = 10,   # Stop early if no improvement after 10 rounds
  verbose = 1                   # Print output
)

# Make predictions on the test set
y_pred <- predict(xgb_model, newdata = dtest)

# Convert predictions to factor for comparison
y_pred <- as.factor(y_pred)
y_test <- as.factor(y_test)

# Calculate metrics for all classes
metrics <- confusionMatrix(y_pred, y_test, mode = "everything")
print(metrics)

# Extract precision, recall, F1-score, and other metrics
precision <- metrics$byClass[, "Precision"]
recall <- metrics$byClass[, "Recall"]
f1_score <- metrics$byClass[, "F1"]

# Print the metrics
cat("Precision by Class: ", precision, "\n")
cat("Recall by Class: ", recall, "\n")
cat("F1-score by Class: ", f1_score, "\n")

# Print the overall metrics like accuracy, Kappa, etc.
cat("Overall Accuracy: ", metrics$overall['Accuracy'], "\n")
cat("Overall Kappa: ", metrics$overall['Kappa'], "\n")

# Get feature importance and plot
importance_matrix <- xgb.importance(model = xgb_model)
xgb.plot.importance(importance_matrix)


###weighted oversampling data model

# Get the class frequencies from the training set
class_freq <- table(y_train)

# Calculate the inverse of frequencies as class weights
class_weights <- max(class_freq) / class_freq

# Print the class weights for reference
print(class_weights)


# Adjust the parameters to include class weights
params <- list(
  objective = "multi:softmax",  # Multi-class classification
  eval_metric = "mlogloss",     # Evaluation metric (log loss)
  num_class = 3,                # Number of classes
  eta = 0.3,                    # Learning rate
  max_depth = 6,                # Maximum depth of trees
  subsample = 0.8,              # Row sampling rate
  colsample_bytree = 0.8        # Feature sampling rate
)

# Train the model with class weights applied
xgb_model_weighted <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train = dtrain, eval = dtest),
  early_stopping_rounds = 10,
  weight = class_weights[as.factor(y_train) + 1],  # Apply class weights
  verbose = 1
)


# Make predictions on the test set
y_pred_weighted  <- predict(xgb_model_weighted, newdata = dtest)

# Convert predictions to factor for comparison
y_pred_weighted <- as.factor(y_pred_weighted)
y_test <- as.factor(y_test)

# Calculate metrics for all classes
metrics_weighted_model_2 <- confusionMatrix(y_pred_weighted, y_test, mode = "everything")
print(metrics_weighted_model_2)

# Extract precision, recall, F1-score, and other metrics
precision <- metrics_weighted_model_2$byClass[, "Precision"]
recall <- metrics_weighted_model_2$byClass[, "Recall"]
f1_score <- metrics_weighted_model_2$byClass[, "F1"]

# Print the metrics
cat("Precision by Class: ", precision, "\n")
cat("Recall by Class: ", recall, "\n")
cat("F1-score by Class: ", f1_score, "\n")

# Print the overall metrics like accuracy, Kappa, etc.
cat("Overall Accuracy: ", metrics_weighted_model_2$overall['Accuracy'], "\n")
cat("Overall Kappa: ", metrics_weighted_model_2$overall['Kappa'], "\n")



##########################################


# Assuming you have confusion matrices from both models (initial and oversampled)
# Replace 'metrics' and 'metrics_weighted_model' with your actual confusion matrices

# Create confusion matrices
cm_initial <- as.data.frame(metrics_1$table)
cm_weighted <- as.data.frame(metrics_weighted_model$table)
cm_oversampled <- as.data.frame(metrics$table)
cm_oversampled_weighted <- as.data.frame(metrics_weighted_model_2$table)

# Plot confusion matrix for initial model
plot_cm_initial <- ggplot(data = cm_initial, aes(x = Prediction, y = Reference)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "white", high = "blue") +
  geom_text(aes(label = Freq), vjust = 1) +
  labs(title = "Confusion Matrix (Initial Model)", x = "Predicted Class", y = "True Class") +
  theme_minimal()

# Plot confusion matrix for weighted model
plot_cm_weighted <- ggplot(data = cm_weighted, aes(x = Prediction, y = Reference)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "white", high = "blue") +
  geom_text(aes(label = Freq), vjust = 1) +
  labs(title = "Confusion Matrix (Weighted Model)", x = "Predicted Class", y = "True Class") +
  theme_minimal()

# Plot confusion matrix for oversampled model
plot_cm_oversampled <- ggplot(data = cm_oversampled, aes(x = Prediction, y = Reference)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "white", high = "blue") +
  geom_text(aes(label = Freq), vjust = 1) +
  labs(title = "Confusion Matrix (Oversampled Model)", x = "Predicted Class", y = "True Class") +
  theme_minimal()

# Plot confusion matrix for oversampled and weighted model
plot_cm_oversampled_weighted <- ggplot(data = cm_oversampled_weighted, aes(x = Prediction, y = Reference)) +
  geom_tile(aes(fill = Freq), color = "white") +
  scale_fill_gradient(low = "white", high = "blue") +
  geom_text(aes(label = Freq), vjust = 1) +
  labs(title = "Confusion Matrix (Oversampled and Weighted Model)", x = "Predicted Class", y = "True Class") +
  theme_minimal()

# Arrange both confusion matrix plots side by side
grid.arrange(plot_cm_initial, plot_cm_oversampled,plot_cm_weighted,plot_cm_oversampled_weighted,  ncol = 2)


