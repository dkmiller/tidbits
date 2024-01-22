from dataclasses import dataclass

import inflect
from injector import inject


@inject
@dataclass
class Plural:
    inflector: inflect.engine

    def render(self, entity: str, count: int) -> str:
        if count <= 1:
            return f"{count} {entity}"
        return f"{count} {self.inflector.plural(entity)}"
