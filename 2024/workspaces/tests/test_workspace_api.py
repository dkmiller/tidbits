import time

import pytest


def test_health_endpoint(api):
    api.get("/healthz").raise_for_status()


def test_create_variant(api, variant):
    api.post("/variants/", json=variant).raise_for_status()


def test_get_variant(api, variant):
    api.get(f"/variants/{variant['name']}").raise_for_status()


def test_create_workspace(api, workspace):
    api.post("/workspaces/", json=workspace).raise_for_status()


def test_get_workspace(api, workspace):
    api.get(f"/workspaces/{workspace['id']}").raise_for_status()


@pytest.mark.timeout(30)
def test_wait_for_workspace_to_be_ready(api, workspace):
    status = ""

    while status.lower() != "running":
        time.sleep(3)
        status = (
            api.get(f"/workspaces/{workspace['id']}")
            .raise_for_status()
            .json()["status"]
        )
        print(f"{status=}")


@pytest.mark.timeout(10)
def test_wait_for_workspace_available_via_proxy(proxy, variant):
    success = False
    while not success:
        time.sleep(3)
        response = proxy.get(variant["readiness"])
        success = response.is_success
        print(f"{response.request.url} {response.status_code=} {response.text=}")
    response.raise_for_status()


def test_delete_workspace(api, workspace):
    api.delete(f"/workspaces/{workspace['id']}").raise_for_status()


def test_workspace_no_longer_exists(api, workspace):
    assert api.get(f"/workspaces/{workspace['id']}").status_code == 404


def test_delete_dangling_workspaces(api):
    workspaces = api.get("/workspaces/").raise_for_status().json()
    responses = []
    for workspace in workspaces:
        responses.append(api.delete(f"/workspaces/{workspace['id']}"))

    assert all(r.is_success for r in responses)


def test_delete_dangling_variants(api):
    variants = api.get("/variants/").raise_for_status().json()
    responses = []
    for variant in variants:
        if "-" in variant["name"]:
            responses.append(api.delete(f"/variants/{variant['name']}"))

    assert all(r.is_success for r in responses)
