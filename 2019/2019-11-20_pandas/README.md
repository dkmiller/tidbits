# Pandas

Attempt to learn more details about the Python Pandas library.

## Links

- [pandas: powerful Python data analysis toolkit](https://pandas.pydata.org/pandas-docs/stable/)
- [Intro to data structures](https://pandas.pydata.org/pandas-docs/stable/getting_started/dsintro.html)
- [Comparison with R / R libraries](https://pandas.pydata.org/pandas-docs/stable/getting_started/comparison/comparison_with_r.html)
- [User guide](https://pandas.pydata.org/pandas-docs/stable/user_guide/index.html)
- [IO tools](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html)

## Learnings

Default set of import statements:

```python
import numpy as np
import pandas as pd
```

### Data structures

A series is essentially an array-indexed array: its constructor accepts
`data` and `index` fields.

There are lots of nice ways of constructing `Series` objects, e.g. from
dictionaries or NumPy arrays.

A `DataFrame` is essentially a dictionary mapping column names to
`Series`, i.e. a fundamentally two-dimensional data structure.

Just like `Series`, there are _lots_ of helper constructors for creating
`DataFrame` objects from anything remotely similar.

The equivalent of R's `mutate` operator is the pandas `assign`
extension method, namely 

```python
df.assign(newCol1 = lambda r: r.colA + r.colB,
    newCol2 = lambda r: abs(r.newCol1))
```

There are _many_ ways of selecting row(s), namely by a column, label,
integer location, slice, or boolean vector. A "row" is a `Series` whose
indices are the columns of the underlying `DataFrame` object.

For data frames with identical column names, arithmetic (e,g,. `+`, `*`)
and numpy operations work naturally.

### I/O

You can load a dataframe from pretty much _anything_, via the comprehensive
`read|write_*` methods. Parquet, Excel, etc.&mdash; it's all good!
