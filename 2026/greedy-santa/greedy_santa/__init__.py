from typing import Callable, TypeVar

import polars as pl

T = TypeVar("T")


def create_dunder(method: str):
    def dunder(self, *args, **kwargs):
        args = (a.wrapped if isinstance(a, RowTracer) else a for a in args)
        kwargs = {k: v.wrapped if isinstance(v, RowTracer) else v for k, v in kwargs.items()}
        return RowTracer(wrapped=getattr(self.wrapped, method)(*args, **kwargs))
    return dunder


NAIVELY_TRANSLATED_DUNDERS = [
    "__add__",
    "__eq__",
    "__gt__",
    "__mul__",
    "__pow__",
    "__radd__",
    "__rpow__",
    "__rtruediv__",
    "__sub__",
    "__truediv__"
]


class RowTracer:
    def __init__(self, wrapped: pl.Expr | None) -> None:
        self.wrapped = wrapped

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
        raise NotImplementedError("Truthy/falsy conversions are not supported yet")
    
    def __int__(self):
        # TODO: replace builtins?
        # https://stackoverflow.com/a/61042819
        return RowTracer(self.wrapped.str.to_integer())

    def split(self, sep: str | None = None):
        return RowTracer(self.wrapped.str.split(by=sep))

    def __iter__(self):
        raise NotImplementedError("For-loops not supported yet")

    def __next__(self):
        raise NotImplementedError("For-loops not supported yet")

    # TODO: all of
    # https://www.pythonmorsels.com/every-dunder-method/#cheat-sheet


for dunder in NAIVELY_TRANSLATED_DUNDERS:
    setattr(RowTracer, dunder, create_dunder(dunder))


def compile[T](callable: Callable[[dict], T]) -> pl.Expr:
    fake_row = RowTracer(None)
    fake_result = callable(fake_row)
    return fake_result.wrapped


def initialize():
    # TODO: add a new "map_elements" method that takes a lambda.
    # What about `return_dtype`? Can we infer it now?
    pass
