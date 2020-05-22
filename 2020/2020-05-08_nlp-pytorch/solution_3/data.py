from bidict import bidict
from collections import Counter
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing import Callable, Dict, List


class RawReviews(Dataset):
    '''
    Naive "wrapper" of the Yelp reviews in a Pandas dataset.
    '''

    def __init__(self, path: str, num_reviews: int):
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#line-delimited-json
        dataframes = pd.read_json(path, lines=True, chunksize=num_reviews)
        self.data = dataframes.__next__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        return row.to_dict()


class Vocabulary(object):
    '''
    Handle the conversion of tokens to and from integers.
    '''

    def __init__(self, token_to_index: Dict[str, int] = None, add_unknown: bool = True, unknown_token: str = "<UNK>"):
        if token_to_index is None:
            token_to_index = {}
        self._token_to_index = bidict(token_to_index)
        self.add_unknown = add_unknown
        self.unknown_index = -1

        if self.add_unknown:
            self.unknown_index = self.add_token(unknown_token)

    def add_token(self, token: str) -> int:
        if token in self._token_to_index:
            index = self._token_to_index[token]
        else:
            index = len(self._token_to_index)
            self._token_to_index[index] = token
        return index

    def lookup_token(self, token: str) -> int:
        if self.add_unknown:
            return self._token_to_index.get(token, self.unknown_index)
        else:
            return self._token_to_index[token]

    def lookup_index(self, index: int) -> str:
        if index not in self._token_to_index.inverse:
            raise KeyError(f'Index {index} is not in the vocabulary.')
        return self._token_to_index.inverse[index]

    def __str__(self) -> str:
        return f'<Vocabulary(size={len(self)}>'

    def __len__(self) -> int:
        return len(self._token_to_index)


class ReviewVectorizer(object):
    '''
    Handle one-hot encoding of reviews.
    '''

    def __init__(self, review_vocab: Vocabulary, rating_vocab: Vocabulary, tokenizer: Callable[[str], List[str]]):
        self.review_vocab = review_vocab
        self.rating_vocab = rating_vocab
        self.tokens = tokenizer

    def vectorize(self, text: str) -> np.ndarray:
        one_hot = np.zeros(len(self.review_vocab), dtype=np.float32)

        for token in self.tokens(text):
            one_hot[self.review_vocab.lookup_token(token)] = 1

        return one_hot

    @classmethod
    def from_raw_reviews(cls, raw_reviews: RawReviews,  tokenizer: Callable[[str], List[str]],cutoff: int = 25) -> ReviewVectorizer:
        review_vocab = Vocabulary(add_unknown=True)
        rating_vocab = Vocabulary(add_unknown=False)

        for rating in sorted(set(review_df.rating)):
            rating_vocab.add_token(rating)

        word_counts = Counter()
        for row in raw_reviews:
            review = row['review']
            for word in tokenizer(review):
                word_counts[word] += 1
                
        for word, count in word_counts.items():
            if count > cutoff:
                review_vocab.add_token(word)

        return cls(review_vocab, rating_vocab)


class VectorizedReviews(Dataset):
    def __init__(self, raw_reviews: RawReviews, vectorizer: ReviewVectorizer):
        self.raw_reviews = raw_reviews
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.raw_reviews)

    def __getitem__(self, index: int) -> Dict:
        row = self.raw_reviews[index]
        review_vector = self.vectorizer.vectorize(row['review'])
        rating_index = self.vectorizer.rating_vocab.lookup_token(row['rating'])

        return {
            'x_data': review_vector,
            'y_target': rating_index
        }

    @classmethod
    def load(cls, path: str, num_reviews: int):
        raw_reviews = RawReviews(path, num_reviews)
        return cls(raw_reviews, )
