import socketserver
from random import randint
from uuid import uuid4

import httpx
from pytest import fixture


@fixture(scope="module")
def port() -> int:
    # https://stackoverflow.com/a/61685162
    with socketserver.TCPServer(("localhost", 0), None) as s:
        return s.server_address[1]


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
def health(variant):
    if variant == "jupyterlab":
        return "/api"
    elif variant == "vscode":
        return "/healthz"
    else:
        raise RuntimeError(f"Unknown variant {variant}")


@fixture(scope="module")
def workspace(variant) -> dict:
    return {
        "id": f"{variant}-{uid()}",
        "name": f"{variant}-{uid()}",
        "image_alias": variant,
        "port": randint(1024, 49151),
    }


@fixture(scope="module")
def proxy(workspace):
    return httpx.Client(
        base_url=f"http://localhost:8002/{workspace['id']}/{workspace['port']}"
    )
