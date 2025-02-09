import asyncio
from typing import Sequence

from fastapi import APIRouter, Depends, HTTPException
from fastapi_injector import Injected
from kubernetes_asyncio.client import ApiClient, AppsV1Api, CoreV1Api
from kubernetes_asyncio import utils
from sqlmodel import Session, select

from server.models import Workspace, WorkspaceResponse
from server.k8s import api_client, v1_apps, v1_api, pod_spec


router = APIRouter()


@router.get("/")
def get_workspaces(session: Session = Injected(Session)) -> Sequence[Workspace]:
    return session.exec(select(Workspace)).all()


@router.get("/{id}")
async def get_workspace(
    id: str,
    session: Session = Injected(Session),
    v1: CoreV1Api = Depends(v1_api),
) -> WorkspaceResponse:
    workspace = session.get(Workspace, id)
    if not workspace:
        raise HTTPException(status_code=404, detail=f"Workspace '{id}' not found")
    ret = await v1.list_pod_for_all_namespaces()
    pods = [p for p in ret.items if p.metadata.labels.get("app") == workspace.id]
    if pods:
        status = pods[0].status.phase
    else:
        status = "unknown"
    rv = WorkspaceResponse.model_validate(workspace, update={"status": status})
    return rv


# TODO: app.put for "start"?
# https://softwareengineering.stackexchange.com/a/388621


@router.post("/")
async def create_workspace(
    workspace: Workspace,
    session: Session = Injected(Session),
    api: ApiClient = Depends(api_client),
) -> Workspace:
    if session.get(Workspace, workspace.id):
        raise HTTPException(
            status_code=409, detail=f"Workspace '{workspace.id}' already exists"
        )
    session.add(workspace)
    manifest, namespace = pod_spec(workspace)
    objects = await utils.create_from_dict(api, manifest, namespace=namespace)
    print(f"Created {len(objects)} objects")

    session.commit()
    session.refresh(workspace)
    return workspace


@router.delete("/{id}")
async def delete_workspace(
    id: str,
    session: Session = Injected(Session),
    v1: CoreV1Api = Depends(v1_api),
    apps_api: AppsV1Api = Depends(v1_apps),
):
    workspace = session.get(Workspace, id)
    if not workspace:
        raise HTTPException(status_code=404, detail=f"Workspace '{id}' not found")
    session.delete(workspace)
    # TODO: more consistent way of getting workspace pod and service IDs.
    await asyncio.gather(
        # https://stackoverflow.com/a/74642309
        apps_api.delete_namespaced_deployment(
            f"{workspace.id}-deployment", namespace="default"
        ),  # type: ignore
        v1.delete_namespaced_service(
            f"{workspace.id}-service",
            namespace="default",
            propagation_policy="Foreground",
        ),  # type: ignore
    )
    session.commit()
    return {"ok": True}
