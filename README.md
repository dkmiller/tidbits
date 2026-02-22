# Tidbits

This repository contains a history of musings,
[toy software](https://blog.jsbarretto.com/post/software-is-joy), and other notes.
Since moving  everything into year-by-year folders, the bibliography may have broken 
temporarily. 

## Backlog of ideas

### [Mirrord](https://metalbear.com/mirrord/)

Run local code in remote Kubernetes context. Another way of doing what
[Ptah's `sync`](https://ptah.readthedocs.io/) command tries to do.

### [Polars: compile Python lambda &mapsto; expression](./2026/greedy-santa/README.md)

Conceptually: "PyTorch-style approach for Polars". Allow syntax like

``` python
df.apply_compiled(lambda row: row["a"] * row["b"] + 1)
```

... where under the hood the lambda gets "traced" (kind of like TorchScript) then
compiled to the appropriate Polars expression.

