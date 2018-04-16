library(readr)
library(tidyverse)

path <- "Prove_11_Ensemble/results.csv"

dat_ <- read_csv(path, col_names = c("dataset", "algorithm", "accuracy", "num_trees", "max_feat"))

dat <- dat_ %>% 
  group_by(dataset, algorithm) %>% 
  mutate(iteration = row_number())

dat %>% 
  ggplot() +
  geom_point(aes(x = num_trees, y = accuracy)) +
  geom_line(aes(x = num_trees, y = accuracy)) +
  facet_grid(algorithm~dataset)
 