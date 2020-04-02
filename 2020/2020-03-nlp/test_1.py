import pandas as pd
import pytest
from solutions_1 import compute_co_occurrence_matrix, distinct_words, END_TOKEN, START_TOKEN
from typing import Dict, List


def tokens(fragment: str) -> List[str]:
    return f'{START_TOKEN} {fragment} {END_TOKEN}'.split(' ')


def prepare_corpus(line_separated_content: str):
    fragments = line_separated_content.split('\n')
    result = list(map(tokens, fragments))
    return result


@pytest.mark.parametrize('corpus,expected', [
    ("All that glitters isn't gold\nAll's well that ends well",
     [END_TOKEN, START_TOKEN, 'All', "All's", 'ends', 'glitters', 'gold', "isn't", 'that', 'well'])
])
def test_distinct_words(corpus, expected):
    corpus = prepare_corpus(corpus)
    result = distinct_words(corpus)
    assert (expected, len(expected)) == result


@pytest.mark.parametrize('window_size, corpus,expected_matrix', [
    (1,
     "All that glitters isn't gold\nAll's well that ends well",
     [[0, 0, 0, 0, 0, 0, 1, 0, 0, 1, ],
      [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, ],
      [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, ],
      [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, ],
      [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, ],
      [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, ],
      [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, ],
      [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, ],
      [0, 0, 1, 0, 1, 1, 0, 0, 0, 1, ],
      [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, ]])
])
def test_compute_co_occurrence_matrix(window_size, corpus, expected_matrix,):
    corpus = prepare_corpus(corpus)
    words, num = distinct_words(corpus)

    m, word_to_index = compute_co_occurrence_matrix(corpus, window_size)
    expected = pd.DataFrame(expected_matrix)

    assert num == len(word_to_index)
    assert expected.shape == m.shape
    assert expected.values.tolist() == m.values.tolist()
