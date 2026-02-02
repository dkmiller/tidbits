from typing import Callable, TypeVar

import polars as pl

T = TypeVar("T")


class RowTracer:
    def __init__(self, wrapped: pl.Expr | None) -> None:
        self.wrapped = wrapped
        # Map "expression identifier / pointer" to raw value.
        # self.expressions: dict[str, pl.Expr] = {}

    def __getitem__(self, key):
        if self.wrapped is None:
            # Get or return a "Tracer" object whose identity is $row[key]
            # self.expressions[f"row[{key}]"] = pl.col(key)
            return RowTracer(wrapped=pl.col(key))
        else:
            if isinstance(key, RowTracer):
                raise NotImplementedError("getitem")
            else:
                if isinstance(key, int):
                    # TODO: we should have a more reliable way of detecting "list"-type.
                    return RowTracer(wrapped=self.wrapped.list.get(key))
                else:
                    raise NotImplementedError("getitem")

    # TODO: how many of
    # https://docs.pola.rs/api/python/dev/reference/expressions/api/polars.Expr.str.to_integer.html
    # can we do?

    def __bool__(self):
        raise NotImplementedError("bool")
        # TODO: if encountering "conversion to truthy", fork:
        # - one path "if true", another "if false".

    def __add__(self, right):
        if isinstance(right, RowTracer):
            return RowTracer(self.wrapped + right.wrapped)
        else:
            return NotImplementedError("add")

    def __int__(self):
        # TODO: replace builtins?
        # https://stackoverflow.com/a/61042819
        return RowTracer(self.wrapped.str.to_integer())

    def __sub__(self, right):
        if isinstance(right, RowTracer):
            return RowTracer(self.wrapped - right.wrapped)

    def __mul__(self, right):
        if isinstance(right, RowTracer):
            return RowTracer(self.wrapped * right.wrapped)

    def split(self, sep: str | None = None):
        return RowTracer(self.wrapped.str.split(by=sep))

        # pl.col("variable").str.split(by="_").list.get(1).alias("row"),



    # TODO: all of
    # https://www.pythonmorsels.com/every-dunder-method/#cheat-sheet


def compile[T](callable: Callable[[dict], T]) -> pl.Expr:
    fake_row = RowTracer(None)
    fake_result = callable(fake_row)
    return fake_result.wrapped


def initialize():
    # TODO: add a new "map_elements" method that takes a lambda.
    # What about `return_dtype`? Can we infer it now?
    pass
