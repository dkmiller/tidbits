import os


def test_key_pair(key_pair):
    assert "private" in key_pair
    assert "public" in key_pair

    assert os.path.isfile(key_pair["private"])
    assert os.path.isfile(key_pair["public"])
