import docker
import logging
import pytest
import requests
from test.common import get_bytes

log = logging.getLogger(__name__)

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

def setup_module():
    client = docker.from_env()
    client.containers.run('danmill/flask', detach=True, ports={5000: 5000})
    containers = client.containers.list()
    log.info(f'Running containers: {containers}')

def teardown_module():
    client = docker.from_env()
    for c in client.containers.list():
        log.info(f'Stopping container {c}')
        c.stop()
