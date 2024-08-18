#Installing necessary packages
install.packages("ggplot2")  # For data visualization
install.packages("dplyr")    # For data manipulation
install.packages("readr")    # For reading data
install.packages("tidyverse") # A collection of essential packages

#Loading Data
customers <- read.csv("F:/University/Projects/Data Science/Statistical Learning/Maven Project/customers.csv")
events <- read.csv("F:/University/Projects/Data Science/Statistical Learning/Maven Project/events.csv")
offers <- read.csv("F:/University/Projects/Data Science/Statistical Learning/Maven Project/offers.csv")

#Data Manipulation

# Assuming your data frame is named df and the column is named 'date_column'
customers$became_member_on <- as.Date(as.character(customers$became_member_on), format = "%Y%m%d")
