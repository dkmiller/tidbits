import os

from jinja2 import Template
from pydantic_yaml import parse_yaml_raw_as

from server.models import Variant, Workspace


class Variants:
    def resolve(self, workspace: Workspace) -> Variant | None:
        template = Template(os.environ["WORKSPACE_VARIANTS"])
        yaml = template.render(port=workspace.port)
        variants: list[Variant] = parse_yaml_raw_as(list[Variant], yaml)
        candidates = [v for v in variants if v.name == workspace.variant]
        return candidates[0] if variants else None
