
import gensim.downloader as api
import nltk
import pandas as pd
import pytest
from solutions_1 import *
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


@pytest.mark.parametrize('window_size,corpus,expected', [
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
def test_compute_co_occurrence_matrix(window_size, corpus, expected,):
    corpus = prepare_corpus(corpus)
    _, num = distinct_words(corpus)

    m, word_to_index = compute_co_occurrence_matrix(corpus, window_size)

    assert num == len(word_to_index)
    assert np.array(expected).shape == m.shape
    assert expected == m.tolist()


@pytest.mark.parametrize('corpus,k', [
    ("All that glitters isn't gold\nAll's well that ends well", 2)
])
def test_reduce_to_k_dim(corpus, k):
    corpus = prepare_corpus(corpus)
    _, num = distinct_words(corpus)

    m, _ = compute_co_occurrence_matrix(corpus, 1)
    m_reduced = reduce_to_k_dim(m, k)

    assert (num, k) == m_reduced.shape


@pytest.mark.parametrize('m_reduced,word_to_index', [
    ([[1, 1], [-1, -1], [1, -1], [-1, 1], [0, 0]],
     {'test1': 0, 'test2': 1, 'test3': 2, 'test4': 3, 'test5': 4})
])
def test_plot_embeddings(m_reduced, word_to_index):
    m_reduced = np.array(m_reduced)
    words = word_to_index.keys()
    plot_embeddings(m_reduced, word_to_index, words)


def read_corpus(reuters_category: str):
    '''
    Read files from the specified Reuter's category.
    '''
    nltk.download('reuters')
    files = nltk.corpus.reuters.fileids(reuters_category)
    return [[START_TOKEN] + [w.lower() for w in list(nltk.corpus.reuters.words(f))] + [END_TOKEN] for f in files]


@pytest.mark.skip(reason='This test takes 2+ minutes to run.')
@pytest.mark.parametrize('corpus_category,words', [
    ('crude', ['barrels', 'bpd', 'ecuador', 'energy', 'industry',
               'kuwait', 'oil', 'output', 'petroleum', 'venezuela'])
])
def test_plot_co_occurrence(corpus_category, words):
    corpus = read_corpus(corpus_category)
    plot_co_occurrence(corpus, words)


@pytest.mark.parametrize('glove_dataset,corpus_size,words', [
    ('glove-wiki-gigaword-200', 10000, ['barrels', 'bpd', 'ecuador', 'energy', 'industry',
                                        'kuwait', 'oil', 'output', 'petroleum', 'venezuela'])
])
def test_plot_glove_embeddings(glove_dataset, corpus_size, words):
    glove_vectors = api.load(glove_dataset)
