'''
All ML and data related code.
'''

import io
import torchvision.transforms as transforms
from PIL import Image

_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225])
])

def transform_image(bytes):
    '''
    Transform an image, encoded as a series of bytes, into a tensor.
    '''
    image = Image.open(io.BytesIO(bytes))
    transormed = _transforms(image)
    return transormed.unsqueeze(0)
