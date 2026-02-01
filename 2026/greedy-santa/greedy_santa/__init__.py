from typing import Callable, TypeVar
from unittest.mock import MagicMock

import polars as pl

T = TypeVar("T")


class RowTracer:
    def __init__(self, wrapped: pl.Expr | None) -> None:
        self.wrapped = wrapped
        # Map "expression identifier / pointer" to raw value.
        # self.expressions: dict[str, pl.Expr] = {}

    def __getitem__(self, key):
        if self.identity is None:
            # Get or return a "Tracer" object whose identity is $row[key]
            # self.expressions[f"row[{key}]"] = pl.col(key)
            return RowTracer(identity=pl.col(key))
        else:
            raise NotImplementedError()
        
    def __bool__(self):
        raise NotImplementedError("bool")
        # TODO: if encountering "conversion to truthy", fork:
        # - one path "if true", another "if false".

    # TODO: all of
    # https://www.pythonmorsels.com/every-dunder-method/#cheat-sheet

def compile[T](callable: Callable[[dict], T]) -> pl.Expr:
    fake_row = RowTracer()
    fake_result = callable(fake_row)

    return pl.col("c")


def initialize():
    # TODO: add a new "map_elements" method that takes a lambda.
    # What about `return_dtype`? Can we infer it now?
