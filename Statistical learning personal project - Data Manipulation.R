#Installing necessary packages
install.packages("dplyr")    # For data manipulation
install.packages("readr")    # For reading data
install.packages("tidyverse") # A collection of essential packages

#Loading Libraries
library(dplyr)
library(tidyr)

#Loading Data
customers <- read.csv("F:/University/Projects/Data Science/Statistical Learning/Maven Project/customers.csv")
events <- read.csv("F:/University/Projects/Data Science/Statistical Learning/Maven Project/events.csv")
offers <- read.csv("F:/University/Projects/Data Science/Statistical Learning/Maven Project/offers.csv")

# Inspect the structure of the data
str(offers)
str(customers)
str(events)

# Preview the first few rows
head(offers)
head(customers)
head(events)

#Data Manipulation

# Extract offer_id from value where it exists
events <- events %>%
  mutate(offer_id = ifelse(grepl("'offer_id':", value),
                           sub(".*'offer_id': '([^']+)'.*", "\\1", value),
                           ifelse(grepl("'offer id':", value),
                                  sub(".*'offer id': '([^']+)'.*", "\\1", value),
                                  NA)))

# Extract reward from value for offer completed events
events <- events %>%
  mutate(reward = ifelse(event == "offer completed",
                         as.numeric(sub(".*'reward': ([0-9]+).*", "\\1", value)),
                         NA))

# Extract amount from value where it exists
events <- events %>%
  mutate(amount = ifelse(event == "transaction",
                         as.numeric(sub(".*'amount': ([0-9.]+).*", "\\1", value)),
                         NA))


# Extract offer_id from the events table
events <- events %>%
  mutate(offer_id = ifelse(event %in% c("offer received", "offer viewed", "offer completed"),
                           sapply(value, function(x) as.numeric(gsub(".*offer_id': ([0-9]+).*", "\\1", x))),
                           NA))


# type converting col date
customers$became_member_on <- as.Date(as.character(customers$became_member_on), format = "%Y%m%d")

# Merge events with customers on customer_id
events_customers <- merge(events, customers, by = "customer_id", all.x = TRUE)

# Merge the combined table with offers on offer_id
final_data <- merge(events_customers, offers, by = "offer_id", all.x = TRUE)

# Ensure the data is ordered by customer and time
final_data <- final_data %>%
  arrange(customer_id, time)

# Fill the existing `offer_id` for "transaction" events
final_data <- final_data %>%
  group_by(customer_id) %>%
  mutate(
    # Further check that the previous event was "offer completed"
    offer_id = ifelse(event == "transaction" & is.na(offer_id) &
                        lag(event, 1) == "offer completed",
                      lag(offer_id, 1), offer_id)
  ) %>%
  ungroup()



###Feature Engineering


# Calculate the number of offers received, viewed, and completed by each customer
customer_events_summary <- final_data %>%
  group_by(customer_id) %>%
  summarise(
    num_offers_received = sum(event == "offer received", na.rm = TRUE),
    num_offers_viewed = sum(event == "offer viewed", na.rm = TRUE),
    num_offers_completed = sum(event == "offer completed", na.rm = TRUE)
  )%>%
  mutate(rate_completed_by_received = num_offers_completed / num_offers_received,
         rate_viewed = num_offers_viewed / num_offers_received
         )

# Create lagged variables to capture the time elapsed between receiving and completing offers
final_data <- final_data %>%
  group_by(customer_id, offer_id) %>%
  mutate(time_received = ifelse(event == "offer received", time, NA),
         time_completed = ifelse(event == "offer completed", time, NA)) %>%
  fill(time_received, .direction = "down") %>%
  fill(time_completed, .direction = "up") %>%
  mutate(time_elapsed = ifelse(!is.na(time_received) & !is.na(time_completed),
                               time_completed - time_received, NA)) %>%
  ungroup()

# Aggregate the time elapsed for each customer
customer_offer_time_summary <- final_data %>%
  group_by(customer_id) %>%
  summarise(avg_time_elapsed = mean(time_elapsed, na.rm = TRUE))

# Calculate total spending and number of transactions
customer_spending_summary <- final_data %>%
  filter(event == "transaction") %>%
  group_by(customer_id) %>%
  summarise(
    total_spent = sum(amount, na.rm = FALSE),
    num_transactions = n()
  ) %>%
  mutate(avg_spending_per_transaction = total_spent / num_transactions)

# Recency: Time since the last transaction
recency_summary <- final_data %>%
  filter(event == "transaction") %>%
  group_by(customer_id) %>%
  summarise(last_transaction_time = max(time)) %>%
  mutate(recency = max(final_data$time) - last_transaction_time)


###Unsupervised Learning Final Dataset

#merging tables
unsupervised <- customers %>%
  full_join(customer_events_summary, by = "customer_id") %>%
  full_join(customer_offer_time_summary, by = "customer_id") %>%
  full_join(customer_spending_summary, by = "customer_id")

#replacing NAN values with 0
unsupervised <- unsupervised %>%
  mutate(across(c(total_spent, num_transactions, avg_spending_per_transaction, rate_completed_by_received, rate_viewed, avg_time_elapsed), ~ replace(., is.na(.), 0)))

#handling null values
#removing null values
unsupervised <- unsupervised[!is.na(unsupervised$income), ]

#save unsupervised dataset
write.csv(unsupervised, file = "F:/University/Projects/Data Science/Statistical Learning/Maven Project/unsupervised.csv", row.names = FALSE)

# Check for missing values
sapply(unsupervised, function(x) sum(is.na(x)))
# Show rows with NA in any column
rows_with_na <- unsupervised %>%
  filter(is.na(rate_completed_by_received) | is.na(rate_completed_by_viewed))

# View the rows
print(rows_with_na)
