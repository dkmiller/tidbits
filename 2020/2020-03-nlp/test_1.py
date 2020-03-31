import pytest
from solutions_1 import distinct_words, END_TOKEN, START_TOKEN


@pytest.mark.parametrize('corpus,expected', [
    ("All that glitters isn't gold\nAll's well that ends well",
     [END_TOKEN, START_TOKEN, 'All', "All's",
      'ends', 'glitters', 'gold',   "isn't",  'that', 'well'])
])
def test_distinct_words(corpus, expected):
    corpus = corpus.split('\n')
    corpus = list(
        map(lambda s: f'{START_TOKEN} {s} {END_TOKEN}'.split(' '), corpus))
    result = distinct_words(corpus)
    assert (expected, len(expected)) == result
