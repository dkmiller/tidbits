import lightgbm as lgb
import os
import pandas as pd
import pathlib
import requests
from sklearn.model_selection import train_test_split
from typing import Tuple


def data_directory() -> str:
    '''
    Common directory for all testing data.
    '''
    pwd = pathlib.Path(__file__).parent.absolute()
    return os.path.join(pwd, '..', '.data')


def gbm_train_test_split(df: pd.DataFrame, target: str, ratio: float = 0.1) -> Tuple[lgb.Dataset, lgb.Dataset]:
    # https://stackoverflow.com/a/32011969

    train, test = train_test_split(df, test_size=ratio)
    pass


def path_in_data_directory(filename: str, url: str) -> str:
    '''
    Download if not already, and if already exists return a working path to
    the file version of the URL.
    '''
    directory = data_directory()
    result = os.path.join(directory, filename)
    if not os.path.exists(result):
        os.makedirs(directory, exist_ok=True)
        r = requests.get(url)
        with open(result, 'wb') as f:
            f.write(r.content)
    return result
