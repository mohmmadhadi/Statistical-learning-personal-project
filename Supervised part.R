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

# Drop the original 'gender' column
supervised$gender <- NULL


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

# Drop the original 'became_member_on' column
supervised$became_member_on <- NULL

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

# Drop the original 'age' column
supervised$age <- NULL

# Drop the original 'income' column
supervised$income <- NULL

# Ordinal encoding for 'age_group'
supervised$age_group <- factor(supervised$age_group, ordered = TRUE,
                       levels = c("young adult", "middle-aged", "senior", "elder"))

# Ordinal encoding for 'income_group'
supervised$income_group <- factor(supervised$income_group, ordered = TRUE,
                          levels = c("low", "medium", "high"))

str(supervised)


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

supervised[is.na(supervised)] <- 0


# Check the distribution of the target variable
table(supervised$preferred_offer_type)

# Convert preferred_offer_type to a factor
supervised$preferred_offer_type <- as.factor(supervised$preferred_offer_type)



###MODEL
# Install the necessary packages
install.packages("xgboost")
install.packages("caret")
install.packages("Matrix")
install.packages("dplyr")

# Load the libraries
library(xgboost)
library(caret)
library(Matrix)
library(dplyr)

# Remove the customer_id as it is not a feature for prediction
supervised <- supervised %>% select(-customer_id)

# Convert the data frame into a matrix, which is required by XGBoost
x_train <- model.matrix(preferred_offer_type ~ . -1, data = train_set)
x_test  <- model.matrix(preferred_offer_type ~ . -1, data = test_set)

# The target variable needs to be converted to numeric (XGBoost needs labels as integers)
y_train <- as.numeric(train_set$preferred_offer_type) - 1 # XGBoost requires 0-based indexing
y_test  <- as.numeric(test_set$preferred_offer_type) - 1

# Convert the dataset into DMatrix objects (used by XGBoost)
dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest  <- xgb.DMatrix(data = x_test, label = y_test)
