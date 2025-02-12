from typing import Sequence

from fastapi import APIRouter, HTTPException, status
from fastapi_injector import Injected
from server.models import Variant
from sqlmodel import Session, select

router = APIRouter()


@router.get("/")
def get_variants(session: Session = Injected(Session)) -> Sequence[Variant]:
    # https://sqlmodel.tiangolo.com/tutorial/select/#create-a-select-statement
    return session.exec(select(Variant)).all()


@router.get("/{name}")
def get_variant(name: str, session: Session = Injected(Session)) -> Variant:
    if not (variant := session.get(Variant, name)):
        raise HTTPException(status_code=404, detail=f"Variant '{variant}' not found")

    return variant


@router.post("/")
def create_variant(
    variant: Variant,
    session: Session = Injected(Session),
):
    if session.get(Variant, variant.name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Variant '{variant.name}' already exists",
        )
    session.add(variant)
    session.commit()
    return variant


@router.delete("/{name}")
def delete_variant(
    name: str,
    session: Session = Injected(Session),
):
    if not (variant := session.get(Variant, name)):
        raise HTTPException(status_code=404, detail=f"Variant '{name}' not found")

    session.delete(variant)
    session.commit()
    return {"ok": True}
