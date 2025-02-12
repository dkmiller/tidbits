from typing import Sequence

from fastapi import APIRouter
from fastapi_injector import Injected
from server.models import Variant
from server.variants import Variants

router = APIRouter()


@router.get("/")
def get_variants(variants: Variants = Injected(Variants)) -> Sequence[Variant]:
    return variants.all()


# TODO: CRUD for variants?
