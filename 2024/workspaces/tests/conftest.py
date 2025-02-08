from random import randint
from uuid import uuid4

import httpx
from pytest import fixture


def uid():
    """
    Short unique string ID.
    """
    return str(uuid4())[:7]


@fixture(scope="module")
def api():
    return httpx.Client(base_url="http://localhost:8000")


# Imitate:
# https://github.com/dkmiller/tidbits/blob/facb960704671729abfc361284d7a017bc2054a9/2023/ssh/conftest.py#L100
@fixture(params=["jupyterlab", "vscode"], scope="module")
def variant(request) -> str:
    return request.param


@fixture(scope="module")
def workspace(variant) -> dict:
    return {
        "id": f"{variant}-{uid()}",
        "name": f"{variant}-{uid()}",
        "image_alias": variant,
        "port": randint(1024, 49151),
    }
