from contextlib import asynccontextmanager
from typing import Sequence

from fastapi import Depends, FastAPI, HTTPException
from fastapi_injector import Injected
from kubernetes_asyncio.client import ApiClient, CoreV1Api
from kubernetes_asyncio import utils
from sqlmodel import Session, select

from server.db import create_db_and_tables
from server.injection import attach
from server.models import Workspace, WorkspaceResponse
from server.k8s import api_client, v1_api, pod_spec


# https://fastapi.tiangolo.com/advanced/events/#lifespan-function
@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield


app = FastAPI(debug=True, lifespan=lifespan)
attach(app)


@app.get("/healthz")
def healthz():
    return "ok"


# TODO: app.put for "start"?
# https://softwareengineering.stackexchange.com/a/388621


@app.get("/workspaces")
def get_workspaces(session: Session = Injected(Session)) -> Sequence[Workspace]:
    return session.exec(select(Workspace)).all()


@app.get("/workspaces/{id}")
async def get_workspace(
    id: str,
    session: Session = Injected(Session),
    v1: CoreV1Api = Depends(v1_api),
) -> WorkspaceResponse:
    workspace = session.get(Workspace, id)
    if not workspace:
        raise HTTPException(status_code=404, detail=f"Workspace '{id}' not found")
    ret = await v1.list_pod_for_all_namespaces()
    pod = [p for p in ret.items if p.metadata.name == workspace.id][0]
    rv = WorkspaceResponse.model_validate(
        workspace, update={"status": pod.status.phase}
    )
    return rv


@app.post("/workspaces/")
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
    print(objects)

    session.commit()
    session.refresh(workspace)
    return workspace


@app.delete("/workspaces/{id}")
async def delete_workspace(
    id: str,
    session: Session = Injected(Session),
    v1: CoreV1Api = Depends(v1_api),
):
    workspace = session.get(Workspace, id)
    if not workspace:
        raise HTTPException(status_code=404, detail=f"Workspace '{id}' not found")
    session.delete(workspace)
    await v1.delete_namespaced_pod(workspace.id, namespace="default")
    session.commit()
    return {"ok": True}
