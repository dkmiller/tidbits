import hashlib
import pytest
import requests
import subprocess
from test.common import get_bytes

@pytest.mark.parametrize('postfix,prediction', [
    ('773/200/200', ['n09472597', 'volcano']),
    ('774/200/200', ['n09332890', 'lakeside'])
    ])
def test_predict_endpoint(postfix, prediction):
    url = f'https://picsum.photos/id/{postfix}.jpg'
    data = get_bytes(url)
    r = requests.post('http://localhost:5000/predict', files={'file': data})
    r_json = r.json()
    class_id = r_json['class_id']
    class_name = r_json['class_name']
    assert [class_id, class_name] == prediction
