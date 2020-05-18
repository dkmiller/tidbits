import pandas as pd
from torch.utils.data import Dataset
from typing import Dict


class RawYelpReviews(Dataset):
    '''
    TODO: docstring.
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


class VectorizedReviews(Dataset):
    def __init__(self, raw_yelp_reviews: RawYelpReviews):
        pass
