'''
Exploring word vectors. A set of utility methods centered around creating and
manipulating word vectors. Based on:

http://web.stanford.edu/class/cs224n/assignments/a1_preview/exploring_word_vectors.html
'''

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from typing import Dict, Iterable, List


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


def compute_co_occurrence_matrix(corpus: Iterable[List[str]], window_size: int = 4) -> (np.ndarray, Dict[str, int]):
    '''
    1.2. Compute the co-occurrence matrix for the given corpus and window size.
    Run time is `O(size of corpus x window_size)`.
    '''
    words, num_words = distinct_words(corpus)
    m = pd.DataFrame(index=words, columns=words).fillna(0)
    word_to_index = {words[i]: i for i in range(num_words)}

    for fragment in corpus:
        s = set()
        fragment_size = len(fragment)
        for window in range(1, window_size + 1):
            for i in range(0, fragment_size - window):
                w1 = fragment[i]
                w2 = fragment[i + window]
                if w1 != w2:
                    s.add((w1, w2))
                    s.add((w2, w1))
        for (w1, w2) in s:
            m[w1][w2] += 1

    return m.to_numpy(), word_to_index


def reduce_to_k_dim(m: np.ndarray, k:int=2) -> np.ndarray:
    '''
    1.3. Produce k-dimensional embeddings using `m`.
    '''
    svd = TruncatedSVD(n_components=k, n_iter=10, random_state=42)
    return svd.fit_transform(m)
