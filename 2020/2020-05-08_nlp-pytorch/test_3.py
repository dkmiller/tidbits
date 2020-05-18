import pytest
from solution_3.data import RawYelpReviews


@pytest.mark.parametrize('file,num_rows', [
    (r'C:\src\tidbits\2020\2020-05-08_nlp-pytorch\yelp-dataset\yelp_academic_dataset_review.json',
     10)
])
def test_raw_yelp_reviews(file: str, num_rows: int):
    dataset = RawYelpReviews(file, num_rows)
    assert len(dataset) == num_rows
    assert all(['text' in d and 'stars' in d for d in dataset])
