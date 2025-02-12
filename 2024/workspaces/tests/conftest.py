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
def variant(request) -> str:
    return request.param


@fixture(scope="module")
def health(variant):
    match variant:
        case "code":
            return "/healthz"
        case "jupyter":
            return "/api"
        case _:
            raise RuntimeError(f"Unknown variant {variant}")


@fixture(scope="module")
def workspace(variant) -> dict:
    return {
        "id": f"{variant}-{uid()}",
        "variant": variant,
    }


@fixture(scope="module")
def proxy(api, workspace):
    variants = api.get("/variants/").raise_for_status().json()
    variant = [v for v in variants if v["name"] == workspace["variant"]][0]
    return httpx.Client(
        base_url=f"http://localhost:8002/workspaces/{workspace['id']}/{variant['ports'][0]}"
    )
