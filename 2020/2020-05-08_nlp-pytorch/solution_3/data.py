from bidict import bidict
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typeguard import typechecked
from typing import Callable, Dict, List


@typechecked
def raw_reviews(path: str, num_reviews: int) -> pd.DataFrame:
    '''
    Naive "wrapper" of the Yelp reviews in a Pandas dataset.
    '''
    dataframes = pd.read_json(path, lines=True, chunksize=num_reviews)
    return dataframes.__next__()


class VectorizedReviews(Dataset):
    def __init__(self, raw_reviews: pd.DataFrame, max_features: int):
        self.raw_reviews = raw_reviews
        self.raw_reviews['rating'] = self.raw_reviews.apply(lambda r: 'positive' if r['stars'] > 2 else 'negative', axis=1)
        self.text_vectorizer = CountVectorizer(max_features=max_features)
        self.text_vectorizer.fit(raw_reviews.text)
        self.rating_vectorizer = CountVectorizer()
        self.rating_vectorizer.fit(raw_reviews.rating)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self) -> int:
        return len(self.raw_reviews)


    @typechecked
    def __getitem__(self, index: int) -> Dict:
        # https://stackoverflow.com/a/47604605/2543689
        row = self.raw_reviews.iloc[index]
        v_text = self.text_vectorizer.transform([row.text]).toarray()
        rating = self.rating_vectorizer.vocabulary_[row.rating]

        return {
            'x_data': self.transform(v_text),
            'y_target': rating
        }
