from bidict import bidict
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset
from typing import Callable, Dict, List


def raw_reviews(path: str, num_reviews: int) -> pd.DataFrame:
    '''
    Naive "wrapper" of the Yelp reviews in a Pandas dataset.
    '''
    dataframes = pd.read_json(path, lines=True, chunksize=num_reviews)
    return dataframes.__next__()


class VectorizedReviews(Dataset):
    def __init__(self, raw_reviews: pd.DataFrame):
        self.raw_reviews = raw_reviews
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(raw_reviews.text)

    def __len__(self):
        return len(self.raw_reviews)

    def __getitem__(self, index: int) -> Dict:
        row = self.raw_reviews[index]
        review_vector = self.vectorizer.transform(row.text)
        rating_index = row.rating

        return {
            'x_data': review_vector,
            'y_target': rating_index
        }
