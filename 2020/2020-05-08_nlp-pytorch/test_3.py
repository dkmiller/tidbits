import pytest
from solution_3.data import RawReviews, VectorizedReviews


@pytest.mark.parametrize('file,num_rows', [
    ('yelp-dataset/yelp_academic_dataset_review.json', 10)
])
def test_raw_reviews(file: str, num_rows: int):
    dataset = RawReviews(file, num_rows)
    assert len(dataset) == num_rows
    assert all(['text' in d and 'stars' in d for d in dataset])
