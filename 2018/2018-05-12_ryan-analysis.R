library(tidyverse)

chapter.lengths <- read_csv('2018-05-12_chapter-lengths.csv', skip=1) %>%
  mutate(chapter = row_number())

pain.sayings <- read_csv('2018-05-12_pain-verses.csv', col_names = FALSE) %>%
  separate(X1, c('chapter', 'verse'), ':') %>%
  mutate(chapter = as.numeric(chapter), verse = as.numeric(verse))

pain.sayings %>%
  left_join(chapter.lengths, by = 'chapter')
