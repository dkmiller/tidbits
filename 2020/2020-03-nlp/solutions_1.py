from typing import List


START_TOKEN = '<START>'
END_TOKEN = '<END>'


def distinct_words(corpus: List[List[str]]) -> (List[str], int):
    '''
    Determine a list of distinct words for the corpos. Returns sorted list of
    distinct words, together with the number of distinct words.
    '''

    # Cheat, follow https://stackoverflow.com/a/2151553 .

    unique_words = set().union(*corpus)
    unique_words = sorted(unique_words)
    num_words = len(unique_words)

    return unique_words, num_words
