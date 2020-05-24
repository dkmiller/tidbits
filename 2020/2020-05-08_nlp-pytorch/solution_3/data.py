from bidict import bidict
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from typing import Callable, Dict, List


def raw_reviews(path: str, num_reviews: int) -> pd.DataFrame:
    '''
    Naive "wrapper" of the Yelp reviews in a Pandas dataset.
    '''
    dataframes = pd.read_json(path, lines=True, chunksize=num_reviews)
    return dataframes.__next__()


class VectorizedReviews(Dataset):
    def __init__(self, raw_reviews: pd.DataFrame, max_features: int):
        self.raw_reviews = raw_reviews
        self.vectorizer = CountVectorizer(max_features=max_features)
        self.vectorizer.fit(raw_reviews.text)
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.raw_reviews)

    def __getitem__(self, index: int) -> Dict:
        # https://stackoverflow.com/a/47604605/2543689
        row = self.raw_reviews.iloc[index]
        transformed = self.vectorizer.transform([row.text]).toarray()
        review_vector = transformed
        rating_index = int(row.stars)

        return {
            'x_data': self.transform(review_vector),
            'y_target': rating_index
        }
