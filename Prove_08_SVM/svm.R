library(e1071)
library(readr)
library(ggplot2)
library(tidyverse)
library(dplyr)

vowel_dat <- read_csv("Prove 08 - SVM/data/vowel.csv")
letters_dat <- read_csv("Prove 08 - SVM/data/letters.csv")

c_values <- c(10^-5, 10^-3, 10^-1, 10^1, 10^3, 10^5, 10^7, 10^9, 10^11, 10^13, 10^15)
gamma_values <- c(10^-15, 10^-13, 10^-11, 10^-9, 10^-7, 10^-5, 10^-3, 10^-1, 10^1, 10^3, 10^5)

# Create grid
grid <- list(cost = c_values, gamma = gamma_values) %>% 
  cross_df()

# Accuracy function
compute_accuracy <- function(fit, test_feat, test_labs) {
  predicted <- predict(fit, test_feat)
  mean(predicted == test_labs)
}

######################## VOWEL DATA ###########################
# Split vowel data into train/test
set.seed(245)
n <- nrow(vowel_dat)
v_train_rows <- sample(seq(n), size = .8 * n)
v_train <- vowel_dat[ v_train_rows, ]
v_test  <- vowel_dat[-v_train_rows, ]

get_v_model <- function(gamma, cost) {
  svm(Class~., data = v_train, type = 'C', gamma = gamma, cost = cost)
}

v_grid <- grid %>%
  mutate(fit = pmap(grid, get_v_model), 
         test_accuracy = map_dbl(fit, compute_accuracy, v_test %>% select(-Class), v_test$Class)) %>% 
  arrange(desc(test_accuracy), cost, gamma)


######################## LETTERS DATA ###########################
c_values <- c(10^-3, 10^-1, 10^1, 10^3)
gamma_values <- c(10^-5, 10^-3, 10^-1, 10^1)
grid <- list(cost = c_values, gamma = gamma_values) %>% 
  cross_df()

# Split letters data into train/test
set.seed(245)
n <- nrow(letters_dat)
l_train_rows <- sample(seq(n), size = .8 * n)
l_train <- letters_dat[ l_train_rows, ]
l_test  <- letters_dat[-l_train_rows, ]

# Just to see the status
counter <- 1

# This is the function that is used in parallel
get_l_model <- function(gamma, cost) {
  svm(letter~., data = l_train, type = 'C', gamma = gamma, cost = cost)
  counter
  counter <- counter + 1
}

# Create a dataset that contains the results of every combination in the grid search
l_grid <- grid %>%
  mutate(fit = pmap(grid, get_l_model), 
         test_accuracy = map_dbl(fit, compute_accuracy, l_test %>% select(-letter), l_test$letter)) %>% 
  arrange(desc(test_accuracy), cost, gamma)
