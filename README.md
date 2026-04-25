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
apps. Explore using it with agentic applications; where the agent can write code to
call tools instead of just calling them directly:
[Executable Code Actions Elicit Better LLM Agents](https://arxiv.org/pdf/2402.01030).

### [Polars: compile Python lambda &mapsto; expression](./2026/greedy-santa/README.md)

Conceptually: "PyTorch-style approach for Polars". Allow syntax like

``` python
df.apply_compiled(lambda row: row["a"] * row["b"] + 1)
```

... where under the hood the lambda gets "traced" (kind of like TorchScript) then
compiled to the appropriate Polars expression.

### Sandboxed Python code via Web Assembly

[Pyodide only supports browsers and Node.js](https://github.com/pyodide/pyodide/discussions/5145),
so one option would be that Python calls Node.js (via the CLI) which runs Pyodide.
Another would be the version of Python compiled for
[wasmer-python](https://github.com/wasmerio/wasmer-python). Ideally, [Monty](#monty)-style
APIs, slower execution, but more comprehensive Python language and package ecosystem support.
