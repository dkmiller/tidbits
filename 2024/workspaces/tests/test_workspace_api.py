import time

import pytest


def test_health_endpoint(api):
    api.get("/healthz").raise_for_status()


def test_create_workspace(api, workspace):
    api.post("/workspaces/", json=workspace).raise_for_status()


def test_get_workspace(api, workspace):
    api.get(f"/workspaces/{workspace['id']}").raise_for_status()


@pytest.mark.timeout(60)
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


# TODO: move this "into" workspace-specific health probe.
@pytest.mark.timeout(30)
def test_wait_for_workspace_available_via_proxy(proxy, health):
    success = False
    while not success:
        time.sleep(3)
        success = proxy.get(health).is_success
        print(f"{success=}")


def test_delete_workspace(api, workspace):
    api.delete(f"/workspaces/{workspace['id']}").raise_for_status()


def test_workspace_no_longer_exists(api, workspace):
    assert api.get(f"/workspaces/{workspace['id']}").status_code == 404


def test_delete_dangling_workspaces(api):
    workspaces = api.get("/workspaces").raise_for_status().json()
    responses = []
    for workspace in workspaces:
        responses.append(api.delete(f"/workspaces/{workspace['id']}"))

    assert all(r.is_success for r in responses)
