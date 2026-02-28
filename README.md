# Tidbits

This repository contains a history of musings,
[toy software](https://blog.jsbarretto.com/post/software-is-joy), and other notes.
Since moving  everything into year-by-year folders, the bibliography may have broken 
temporarily. 

## Backlog of ideas

### [gVisor](https://gvisor.dev/docs/)

It _should_ be possible to run sandboxed code inside a Kubernetes + Docker context.

### [Mirrord](https://metalbear.com/mirrord/)

Run local code in remote Kubernetes context. Another way of doing what
[Ptah's `sync`](https://ptah.readthedocs.io/) command tries to do.

### [Monty](https://pydantic.dev/articles/pydantic-monty)

Sandboxed execution of a subset of the Python standard library, excellent for agentic
apps.

### [Polars: compile Python lambda &mapsto; expression](./2026/greedy-santa/README.md)

Conceptually: "PyTorch-style approach for Polars". Allow syntax like

``` python
df.apply_compiled(lambda row: row["a"] * row["b"] + 1)
```

... where under the hood the lambda gets "traced" (kind of like TorchScript) then
compiled to the appropriate Polars expression.

