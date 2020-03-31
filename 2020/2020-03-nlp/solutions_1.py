import torch
from typing import Dict, List


START_TOKEN = '<START>'
END_TOKEN = '<END>'


def distinct_words(corpus: List[List[str]]) -> (List[str], int):
    '''
    1.1. Determine a list of distinct words for the corpos. Returns sorted list
    of distinct words, together with the number of distinct words.
    '''

    # Cheat, follow https://stackoverflow.com/a/2151553 .

    unique_words = set().union(*corpus)
    unique_words = sorted(unique_words)
    num_words = len(unique_words)

    return unique_words, num_words


def compute_co_occurrence_matrix(corpus: List[List[str]], window_size: int = 4) -> (torch.Tensor, Dict[str, int]):
    '''
    1.2. Compute the co-occurrence matrix for the given corpus and window size.
    '''
    words, num_words = distinct_words(corpus)
    M = None
    word2Ind = {}

    return M, word2Ind
