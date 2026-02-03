## Design question: how to handle conditionals?

Suppose we have a Python lambda with an "if / else" statement, for example

``` python
lambda row: "true" if row["col"] == "val" else "false"
```

How should we translate this to a Polars expression _purely based on evaluation_ (i.e.,
no Python AST awareness). It would appear that all we can do is fork execution:

- Let the partially compiled representation of `row["col"] == "val"` be "truthy" in one
  fork, resulting in `"true"`
- Let me be "falsy" in another fork, resulting in `"false"`

Then, stitch the two together with a native Polars conditional:

``` python
pl.when(pl.col("col") == "val").then("true").otherwise("false")
```

As it is, this is fine. However, what to do when the Python method does something with
the output of the conditional, e.g.

``` python
lambda row: (row["true"] if row["col"] == "val" else row["false"]) + ".suffix"
```

The approach :point_up: will "duplicate" the conditional entries:

``` python
pl.when(pl.col("col") == "val").then(pl.col("true") + ".suffix").otherwise(pl.col("false") + ".suffix")
```

It will get _extremely_ messy if there are e.g. conditionals inside for-loops.

``` python
def my_lambda(row):
    counter = 1
    for entry in row["col"].split(":"):
        if entry == "foo":
            counter += 1
    return counter
```

That raises the broader question: should we even support for-loops? Probably not...

---


We are encountering a "fork in the road"!

- Continue execution with a "truthy" response --> capture the result.
- Continue execution with a "falsy" response --> capture the result.

Capture internal tree of boolean switches:

```
()

...

() --> (switch):
    eval  () --> (true)
    queue () --> (false)

...

() --> (true) --> (switch)
    eval  () -> (T) -> (T)

"Double" the queue:
    queue () -> (F) -> (T)
    queue () -> (F) -> (F)
```

Problem: if we "stitch" together when(condition).then(first).otherwise(second),
we invert order of operations.

``` python
2 * ("true" if row["a"] == 1 else "false")
```

should compile to

``` python
2 * pl.when(pl.col("a") == 1).then("true").otherwise("false")
```

not

``` python
pl.when(pl.col("a") == 1).then(2 * "true").otherwise(2 * "false")
```