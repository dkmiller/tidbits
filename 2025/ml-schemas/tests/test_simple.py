from dataclasses import dataclass

from ml_schemas.core._types import Int16


@dataclass
class Schema:
    input_ids: list[Int16]


Schema(input_ids=[1, 2, 3])
