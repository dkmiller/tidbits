import socketserver
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
def api() -> httpx.Client:
    return httpx.Client(base_url="http://localhost:8000")


# Imitate:
# https://github.com/dkmiller/tidbits/blob/facb960704671729abfc361284d7a017bc2054a9/2023/ssh/conftest.py#L100
@fixture(params=["code", "jupyter"], scope="module")
def variant_template(request) -> str:
    return request.param


# Imitate:
# https://github.com/dkmiller/tidbits/blob/facb960704671729abfc361284d7a017bc2054a9/2023/ssh/conftest.py#L100
@fixture(scope="module")
def variant(api, port, variant_template) -> dict:
    definition = api.get(f"/variants/{variant_template}").raise_for_status().json()
    definition["name"] += "-" + uid()
    assert len(definition["ports"]) == 1
    original_port = str(definition["ports"][0])
    definition["ports"] = [port]
    definition["container_args"] = [
        arg.replace(original_port, str(port)) for arg in definition["container_args"]
    ]

    return definition


@fixture(scope="module")
def workspace(variant) -> dict:
    return {
        "id": f"{variant['name']}-{uid()}",
        "variant": variant["name"],
    }


@fixture(scope="module")
def proxy(variant, workspace):
    return httpx.Client(
        base_url=f"http://localhost:8002/workspaces/{workspace['id']}/{variant['ports'][0]}"
    )
