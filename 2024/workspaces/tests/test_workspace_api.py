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


# TODO: test UI is available via the proxy.


# Command line equivalent:
# curl http://localhost:8000/workspaces
def test_delete_workspace(api, workspace):
    api.delete(f"/workspaces/{workspace['id']}").raise_for_status()


def test_delete_dangling_workspaces(api):
    workspaces = api.get("/workspaces").raise_for_status().json()
    responses = []
    for workspace in workspaces:
        responses.append(api.delete(f"/workspaces/{workspace['id']}"))

    assert all(r.is_success for r in responses)
