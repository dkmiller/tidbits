# Greedy Santa: Python lambda &mapsto; Polars expressions

Name: Polars &approx; North Pole &approx; Santa but it's lazy. This is the greedy
version, hence "greedy Santa".

Get started!

``` bash
pip install git+https://github.com/dkmiller/tidbits.git#egg=greedy-santa&subdirectory=2026/greedy-santa
```

... then

``` python
import polars as pl
from greedy_santa import initialize

# Will be accelerated + lazy if possible.
df.map_rows(lambda row: row["a"] * row["b"])
```

## Future

- Regular expressions
- Length (requires "hacking" builtins)
- Override `"lit".split` (more builtin hacking)
- Type-aware casting (e.g. number &times; string should include a cast on the latter)
- Structs?
- Compile TorchScript to Polars expressions?
- Support for [conditionals and/or for-loops](./doc/rfc-0-conditionals.md)
- Programmatically popular all dunder methods in the "obvious" manner.
