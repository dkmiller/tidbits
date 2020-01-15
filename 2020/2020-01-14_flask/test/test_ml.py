import sys
from os import path
from urllib import request

# https://stackoverflow.com/a/27876800
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from src.ml import transform_image

def test_transform_image_on_small_image():
    url = 'https://picsum.photos/200'
    response = request.urlopen(url)
    data = response.read()
    tensor = transform_image(data)
    assert tensor != None
