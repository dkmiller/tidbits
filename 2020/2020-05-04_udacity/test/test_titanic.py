from src import common, titanic
import os
import pandas as pd
import pytest
import requests


@pytest.mark.skip
def test_prepare_data():
    data = titanic.raw_data()
    result = titanic.prepare_data(data)
    assert type(result) == pd.DataFrame


def test_titanic_csv():
    result = titanic.titanic_csv()
    assert result != None


def raw_data():
    result = titanic.raw_data()
    assert type(result) == pd.DataFrame


def setup_module():
    titanic.titanic_csv()
