library(readr)
library(tidyverse)
library(lubridate)

path <- "~/Desktop/hcp_new_new.csv"

dat_ <- read_csv(path)

tester <- dat_ %>% 
  group_by(interviewee_id) %>% 
  filter(1 %in% unique(completed)) %>% 
  ungroup()

# test <- dat_ %>% 
#   filter(new_id == 12260)
part_dat <- sample_n(tester, 10000)
# 
# dat <- dat_ %>%
#   select(interview_id, interviewee_id, dispo_id, dispo_text, vantage_text, created) %>%
#   filter(!vantage_text %in% c("PROOFING", "RECURRING", "BEGIN"), year(created) %in% c(2016, 2017)) %>%
#   mutate(completed = case_when(vantage_text == "COMPLETE" | vantage_text == "PARTIAL" ~ 1, TRUE ~ 0), hour = hour(created)) %>%
#   group_by(interviewee_id, dispo_id, completed, hour) %>%
#   filter(row_number(interview_id) == 1) %>%
#   arrange(interviewee_id)
#
# total <- dat_ %>%
#   select(interview_id, interviewee_id, dispo_id, dispo_text, vantage_text, created) %>%
#   filter(!vantage_text %in% c("PROOFING", "RECURRING", "BEGIN"), year(created) %in% c(2016, 2017)) %>%
#   mutate(completed = case_when(vantage_text == "COMPLETE" | vantage_text == "PARTIAL" ~ 1, TRUE ~ 0), hour = hour(created))
# 
# dat <- dat %>%
#   filter(dispo_id != 0 | vantage_text != "UNABLE")
# 
# test <- dat %>% 
#   filter(!is.na(dispo_id) | vantage_text != "RESUME")
# 
# proof <- test %>% 
#   filter(interview_id == 58699)
# 
# id <- 1
# num <- 1
# tester <- test %>% 
#   ungroup()
# tester["new_id"] <- NA
# tester["num_calls"] <- NA
# # x <- 1625123 
# 
# for (i in x:length(tester$new_id)) {
#   tester$num_calls[i] <- num
#   num <- num + 1
#   tester$new_id[i] <- id
#   if (i %% 100 == 0) {
#     print(c(id, num, i))
#     print(paste(format(round(((1855896-i)/100*2.5/60), 2), nsmall = 2), "minutes left", sep = " "))
#   }
#   if (tester$completed[i] == 1) {
#     id <- id + 1
#     num <- 1
#   }
#   x <- i
# }
# 
# write.csv(tester, file = "~/Desktop/hcp_new_new.csv")

