import pytest
import sys
from os import path
from urllib import request

# https://stackoverflow.com/a/27876800
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from src.ml import transform_image

@pytest.mark.parametrize('postfix', ['300', '800', '/100/300'])
def test_transform_image_on_small_image(postfix):
    url = f'https://picsum.photos/{postfix}'
    response = request.urlopen(url)
    data = response.read()
    tensor = transform_image(data)
    assert len(tensor) == 1
    assert tensor.shape[0] == 1
    assert tensor.shape[1] == 3
    assert tensor.shape[2] == 224
    assert tensor.shape[3] == 224
