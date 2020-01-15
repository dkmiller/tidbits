import pytest
import sys
from os import path
from urllib import request

# https://stackoverflow.com/a/27876800
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from src.ml import get_prediction, transform_image

def get_bytes(url):
    response = request.urlopen(url)
    return response.read()

@pytest.mark.parametrize('postfix,class_id', [('/773/200/200', '980')])
def test_get_prediction(postfix, class_id):
    url = f'https://picsum.photos/id/{postfix}.jpg'
    data = get_bytes(url)
    ŷ = get_prediction(data)
    ŷ = str(ŷ.item())
    assert ŷ == class_id

@pytest.mark.parametrize('postfix', ['300', '800', '/100/300'])
def test_transform_image_on_small_image(postfix):
    url = f'https://picsum.photos/{postfix}'
    data = get_bytes(url)
    tensor = transform_image(data)
    assert len(tensor) == 1
    assert tensor.shape[0] == 1
    assert tensor.shape[1] == 3
    assert tensor.shape[2] == 224
    assert tensor.shape[3] == 224
