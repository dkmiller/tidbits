from contextlib import asynccontextmanager
from typing import Sequence

from fastapi import FastAPI, HTTPException
from fastapi_injector import Injected
from kubernetes import client, config
from sqlmodel import Session, select

from server.db import create_db_and_tables
from server.injection import attach
from server.models import Workspace


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
def get_workspace(id: str, session: Session = Injected(Session)) -> Workspace:
    workspace = session.get(Workspace, id)
    if not workspace:
        raise HTTPException(status_code=404, detail=f"Workspace '{id}' not found")
    return workspace


@app.post("/workspaces/")
def create_workspace(workspace: Workspace, session: Session = Injected(Session)) -> Workspace:
    if session.get(Workspace, workspace.id):
        raise HTTPException(status_code=409, detail=f"Workspace '{workspace.id}' already exists")
    session.add(workspace)
    session.commit()
    session.refresh(workspace)
    return workspace


@app.delete("/workspaces/{id}")
def delete_hero(id: str, session: Session = Injected(Session)):
    workspace = session.get(Workspace, id)
    if not workspace:
        raise HTTPException(status_code=404, detail=f"Workspace '{id}' not found")
    session.delete(workspace)
    session.commit()
    return {"ok": True}


@app.get("/test-k8s")
def k8s():
    # https://github.com/kubernetes-client/python/blob/master/examples/in_cluster_config.py
    config.load_incluster_config()

    v1 = client.CoreV1Api()
    ret = v1.list_pod_for_all_namespaces(watch=False)
    return {"pods": list(map(str, ret.items))}
