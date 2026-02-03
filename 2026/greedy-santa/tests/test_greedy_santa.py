import pytest
import polars as pl

from greedy_santa import compile


@pytest.mark.parametrize(
    "callable,expected",
    [
        (lambda row: row["a"] + row["b"], pl.col("a") + pl.col("b")),
        (lambda row: row["c"] - row["d"], pl.col("c") - pl.col("d")),
        (lambda row: row["e"] * row["f"], pl.col("e") * pl.col("f")),
        (lambda row: row["g"].split(":")[0], pl.col("g").str.split(by=":").list.get(0)),
        (lambda row: row["age"] > 28, pl.col("age") > 28),
        # (lambda row: int(row["h"]), pl.col("h").str.to_integer),
    ],
)
def test_compile(callable, expected):
    compiled = compile(callable)
    assert compiled.meta.serialize() == expected.meta.serialize()


def _for_loop(row):
    for _ in row["list"]:
        return "val"


@pytest.mark.parametrize(
    "callable",
    [
        lambda row: "hi" if row["h"] == "val" else "bye",
        _for_loop,
    ],
)
def test_unsupported(callable):
    with pytest.raises(NotImplementedError):
        compile(callable)
