# Tidbits

This repository contains a history of musings,
[toy software](https://blog.jsbarretto.com/post/software-is-joy), and other notes.
Since moving  everything into year-by-year folders, the bibliography may have broken 
temporarily. 

## Backlog of ideas

### Polars: compile Python lambda &mapsto; expression

Conceptually: "PyTorch-style approach for Polars". Allow syntax like

``` python
df.apply_compiled(lambda row: row["a"] * row["b"] + 1)
```

... where under the hood the lambda gets "traced" (kind of like TorchScript) then compiled to the appropriate Polars expression.
