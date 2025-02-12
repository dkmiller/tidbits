from typing import Sequence

from fastapi import APIRouter, HTTPException
from fastapi_injector import Injected
from server.k8s import K8s
from server.models import Variant, Workspace, WorkspaceResponse
from sqlmodel import Session, select

router = APIRouter()


@router.get("/")
def get_workspaces(session: Session = Injected(Session)) -> Sequence[Workspace]:
    return session.exec(select(Workspace)).all()


@router.get("/{id}")
async def get_workspace(
    id: str,
    session: Session = Injected(Session),
    k8s: K8s = Injected(K8s),
) -> WorkspaceResponse:
    if not (workspace := session.get(Workspace, id)):
        raise HTTPException(status_code=404, detail=f"Workspace '{id}' not found")

    if pod := await k8s.get_pod(workspace):
        status = pod.status.phase
    else:
        status = "unknown"

    return WorkspaceResponse.model_validate(workspace, update={"status": status})


# TODO: app.put for "start"?
# https://softwareengineering.stackexchange.com/a/388621


@router.post("/")
async def create_workspace(
    workspace: Workspace,
    session: Session = Injected(Session),
    k8s: K8s = Injected(K8s),
) -> Workspace:
    if session.get(Workspace, workspace.id):
        raise HTTPException(
            status_code=409, detail=f"Workspace '{workspace.id}' already exists"
        )
    session.add(workspace)

    if not (variant := session.get(Variant, workspace.variant)):
        raise HTTPException(
            status_code=422,
            detail=f"Variant '{workspace.variant}' is not already defined!",
        )

    await k8s.create(workspace, variant)

    session.commit()
    session.refresh(workspace)
    return workspace


@router.delete("/{id}")
async def delete_workspace(
    id: str,
    session: Session = Injected(Session),
    k8s: K8s = Injected(K8s),
):
    if not (workspace := session.get(Workspace, id)):
        raise HTTPException(status_code=404, detail=f"Workspace '{id}' not found")

    session.delete(workspace)
    await k8s.delete(workspace)

    session.commit()
    return {"ok": True}
