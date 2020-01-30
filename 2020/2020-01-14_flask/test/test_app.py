import docker
import logging
import os
import pytest
import requests
import time
from test.common import get_bytes

log = logging.getLogger(__name__)

ENDPOINT_ENV_VAR = 'INFERENCE_ENDPOINT'

def use_local_endpoint():
    return ENDPOINT_ENV_VAR not in os.environ

def get_endpoint():
    return os.getenv(ENDPOINT_ENV_VAR, 'http://localhost:5000')

def test_hello_endpoint():
    endpoint = get_endpoint()
    r = requests.get(endpoint)
    content = r.content
    assert content == b'Hi there!'

@pytest.mark.parametrize('postfix,prediction', [
    ('773/200/200', ['n09472597', 'volcano']),
    ('774/200/200', ['n09332890', 'lakeside'])
    ])
def test_predict_endpoint(postfix, prediction):
    url = f'https://picsum.photos/id/{postfix}.jpg'
    data = get_bytes(url)
    endpoint = get_endpoint()
    r = requests.post(f'{endpoint}/predict', files={'file': data})
    r_json = r.json()
    class_id = r_json['class_id']
    class_name = r_json['class_name']
    assert [class_id, class_name] == prediction

def setup_module():
    if use_local_endpoint():
        client = docker.from_env()
        client.containers.run('danmill/flask', detach=True, ports={5000: 5000})
        seconds = 2
        # Without this, the test_hello_endpoint test starts immediately, before
        # the Flask app is properly running inside Docker.
        log.info(f'Waiting {seconds} seconds for Flask to start.')
        time.sleep(seconds)
        containers = client.containers.list()
        log.info(f'Running containers: {containers}')
    else:
        endpoint = get_endpoint()
        log.info(f'No set up needed, endpoint already up at {endpoint}.')

def teardown_module():
    if use_local_endpoint():
        client = docker.from_env()
        for c in client.containers.list():
            log.info(f'Stopping container {c}')
            c.stop()
    else:
        endpoint = get_endpoint()
        log.info(f'No teardown needed, endpoint already up at {endpoint}.')
