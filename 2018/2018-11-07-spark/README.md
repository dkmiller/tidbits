# Spark

First steps with Apache Spark on a local machine, following
_[Learning Apache Spark 2](https://www.safaribooksonline.com/library/view/learning-apache-spark/9781785885136/)_,
by Muhammad Asif Abbasi, and
_[Scala Cookbook](https://www.safaribooksonline.com/library/view/scala-cookbook/9781449340292/)_,
by Alvin Alexander.

You need the current version of [Docker for Windows](https://docs.docker.com/docker-for-windows/install/)
(&ge; `18.06.1-ce`, installable [via Chocolatey](https://chocolatey.org/packages/docker-desktop))
installed and running.

## Running

Simply run [Run.ps1](./Run.ps1). It will initialize a single-instance Spark cluster
in a minimized PowerShell window and enter that instance's bash prompt as soon as
it is ready.

You can visit the Spark UI at http://localhost:8080, and see the jobs http://localhost:4040/jobs/.

From the Spark window:

```bash
# Start a Spark REPL.
./bin/spark-shell

# Run an example Spark program starting from a pre-existing fat JAR.
./bin/spark-submit \
  --class org.apache.spark.examples.SparkPi \
  --master local[2] \
  examples/jars/spark-examples_2.11-2.3.1.jar 10
```

## Troubleshooting

If `Run.ps1` is hanging, follow [this article](https://docs.docker.com/engine/reference/commandline/system_prune/)
and run

```powershell
docker system prune --force
```

If `docker-spark` isn't syncing, follow [this answer](https://stackoverflow.com/a/1032653) and run:

```powershell
git submodule update --recursive
```

## Notes

Use `:quit` to exit the Scala shell.

### Control Structures

Use [`@switch` annotation](https://stackoverflow.com/q/23985248) when possible.

Multiple cases on one line via `|`.

Generally, `match` statements are awesome!

Try/catch blocks use `match` expressions.

You can create your own control structures, but may want to annotate with `@tailrec`.

### Classes

The body of a class is _actually_ the default constructor.

You can call the generated `_$eq` methods directly.

Case class == record types?

Add constructors to case classes via `apply()` methods. More generally, it looks
like `apply` lets you overload the constructor :open_mouth:.

For private primary constructor, put `private` _after_ class name.

[Companion Objects](https://stackoverflow.com/q/609744) seem like a substitute for
static-style stuff.

Scala supports C\#-style named and default parameters.

Use `propertyName_` to override the setter.

Making a member `private` keeps it from having getters / setters.

[Object-private members](https://alvinalexander.com/scala/how-to-control-scala-method-scope-object-private-package)
(`private[this]`) are "really" private. In C\# and Scala both, private
members can be accessed by other instances of the same class.

Making a `val` member `lazy` means it doesn't get a value assigned until the
value is needed.

Scala does generics via `[ ]`, unlike `< >` for C\#.
