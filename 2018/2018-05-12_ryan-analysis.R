library(tidyverse)
library(stats)

discrepancy<- function(xs) {
  #' Computes the discrepancy of a sequence (values assumed to be in the range
  #' [0,1]).
  ks.test(xs, 'punif')$statistic
}

# Table with chapter numbers and their lengths in verses.
chapter.lengths <- read_csv('2018-05-12_chapter-lengths.csv', skip=1) %>%
  mutate(chapter = row_number())

# Table with, for each chapter, the total number of verses preceeding
# that chapter.
chapter.offsets <- chapter.lengths %>%
  mutate(offset = cumsum(lag(length, default=0)))

# Table with the chapter and verse of each pain saying.
pain.sayings <- read_csv('2018-05-12_pain-verses.csv', col_names = FALSE) %>%
  separate(X1, c('chapter', 'verse'), ':') %>%
  mutate(chapter = as.numeric(chapter), verse = as.numeric(verse))

# Get the index (starting at 1) of the pain sayings.
pain.sayings.indices <- pain.sayings %>%
  left_join(chapter.offsets, by = 'chapter') %>%
  mutate(verse.index = offset + verse)

# Normalize those indices to lie in the interval [0,1].
pain.sayings.normalized <- pain.sayings.indices$verse.index / 915

# The Kolmogorov-Smirnov statistic is 0.24669, which translates to
# p-value = 0.002512.
ks.test(pain.sayings.normalized, 'punif')


## -------- Everything below is (strictly speaking) unnecessary ---------------


# Create a large number of "example sequences of pain sayings", all
# drawn from a uniform distribution.
num.verses <- nrow(pain.sayings)
num.examples <- 100000
examples <- matrix(runif(num.verses * num.examples), num.examples) %>%
  as_tibble

# Compute the discrepancy of all those examples.
discrepancy.examples <- apply(examples, 1, discrepancy)

# Plot a histogram to see the distribution of discrepancies.
discrepancy.examples %>%
  hist(freq=FALSE, breaks = 50)

# Probability of discrepancy as extreme as in the example.
# This comes out to ~ 0.002, which is still well past the "statistically
# significant" cutoff.
mean(discrepancy.examples >= pain.sayings.discrepancy)
