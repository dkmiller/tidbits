import pytest
import polars as pl

from greedy_santa import compile


@pytest.mark.parametrize("callable,expected", [
    (lambda row: row["a"] + row["b"], pl.col("a") + pl.col("b")),
    # (lambda row: "hi" if row else "bye", pl.col("boo")),
])
def test_compile(callable, expected):
    compiled = compile(callable)
    assert compiled.meta.serialize() == expected.meta.serialize()
