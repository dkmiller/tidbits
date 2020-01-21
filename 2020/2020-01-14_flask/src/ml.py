'''
All ML and data related code.
'''

import io
import json
import lazy_object_proxy as lazy
import tempfile
import torchvision.transforms as transforms
import urllib.request as request
from os import path
from PIL import Image
from torchvision import models

def _init_index():
    index_file = path.join(tempfile.gettempdir(), 'imagenet_class_index.json')
    url = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'
    if not path.exists(index_file):
        request.urlretrieve(url, index_file)
    return json.load(open(index_file))

def _init_model():
    model = models.densenet121(pretrained=True)
    model.eval()
    return model

_index = lazy.Proxy(_init_index)
_model = lazy.Proxy(_init_model)

_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])
])

def get_prediction(bytes):
    tensor = transform_image(bytes)
    outputs = _model.forward(tensor)
    _, ŷ = outputs.max(1)
    key = str(ŷ.item())
    return _index[key]

def transform_image(bytes):
    '''
    Transform an image, encoded as a series of bytes, into a tensor.
    '''
    image = Image.open(io.BytesIO(bytes))
    transormed = _transforms(image)
    return transormed.unsqueeze(0)
