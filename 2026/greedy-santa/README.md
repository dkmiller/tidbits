### Polars: compile Python lambda &mapsto; expression

Conceptually: "PyTorch-style approach for Polars". Allow syntax like

``` python
df.apply_compiled(lambda row: row["a"] * row["b"] + 1)
```

... where under the hood the lambda gets "traced" (kind of like TorchScript) then
compiled to the appropriate Polars expression.

Name: Polars &approx; North Pole &approx; Santa but it's lazy. This is the greedy
version, hence "greedy Santa".

Future:

- Compile TorchScript to Polars expressions?
