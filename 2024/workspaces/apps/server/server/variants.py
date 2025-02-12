import os

from fastapi import HTTPException, status
from pydantic_yaml import parse_yaml_raw_as

from server.models import Variant


class Variants:
    # TODO: cache the parsed mapping variant name --> deserialized object?
    def all(self) -> list[Variant]:
        yaml = os.environ["WORKSPACE_VARIANTS"]
        return parse_yaml_raw_as(list[Variant], yaml)  # type: ignore

    def resolve(self, variant: str) -> Variant:
        variants = self.all()
        candidates = [v for v in variants if v.name == variant]
        if candidates:
            return candidates[0]
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Unknown {variant=}")
