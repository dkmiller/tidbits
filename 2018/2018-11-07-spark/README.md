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

Simply run [spark.ps1](./spark.ps1). It will initialize a
single-instance Spark cluster in a minimized PowerShell window and enter
that instance's bash prompt as soon as it is ready.

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

Alternatively, to run a simple Scala script locally, enter

```powershell
.\scala.ps1 Hello.scala arg0 arg1 arg2
```

## Troubleshooting

If `docker-spark` isn't syncing, follow [this answer](https://stackoverflow.com/a/1032653) and run:

```powershell
git submodule update --recursive
```

If you get an error:

> image operating system "linux" cannot be used on this platform.

then follow the instructions [here](https://github.com/docker/kitematic/issues/2696)
to switch Docker to use Linux containers.

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

Class inheritance via `extends`.

Auxilary constructors _can't_ call a superclass constructor.

Scala _does_ support abstract classes _a la_ C\#, but they're discouraged in favor
of traits.

### Methods

Access modifiers are similar to C\#, except `private[packageName]` is used
instead of `internal`.

Tuple types behave just as in C\#, except you _can't_ have named tuple members.

Declare methods without parentheses just as in C\# when declaring properties.

Varargs (like C\# `params`) are declared via `*`.

Annotate that your method will throw an exception via `@throws`.

Scala supports [fluent interfaces](https://www.martinfowler.com/bliki/FluentInterface.html).

Casting is done via `asInstanceOf[T]`, a bit more verbose than C\# or Java.

Reflection via `classOf[T]`.

"What am I" (_a la_ C\#'s `GetType()`) via `getClass`.

### Objects

Scala source code, e.g. [App.scala](https://github.com/scala/scala/blob/v2.12.8/src/library/scala/App.scala)
is easily browsable.

Singleton pattern via `object`.

Package objects seem like a magic way of putting package-specific "magic"
in one place.

### Packages

They're working on [Scala 3](https://www.scala-lang.org/blog/2018/04/19/scala-3.html)
(currently known as [dotty](http://dotty.epfl.ch/)).

You can use `package` like C\#'s `namespace`, or at the top of a file without extra
indentation (but only if there's at most one class in the file).

Imports in three ways, `import a.b.c`, `import a.{b, d}`, and `import a._`.

You can rename imports (nice!) and even class members (:dizzy_face:).

You can hide specific imports via `import a.{b => _, _}`.

Nothing special needed for static imports (_a la_ C\#'s `using static`).

### Traits

High-level: they're like interfaces in C\# and Java.

A trait member without type declarations is assumed to be of type `Unit`,
with no parameters.

Traits (unlike C\# or Java) can have concrete members.

Traits can inherit _anything_ (classes or other traits).

Traits can place restrictions on subtypes via `this: T =>`.

### Functional Programming

In Scala, _everything_ is an expression which yields a value.

Anonymous functions via `=>` just like C\#.

Specify function types by `() => T`, instead of `Function<T>` in C\#.

Generic methods the obvious way via `def exec[T](f:(T) => Unit, t: T) : Unit = f(t)`.

You can curry via the magical `_`.

Scala has ["partial functions"](https://www.scala-lang.org/api/current/scala/PartialFunction.html)
as a commonly used thing.

### Collections

There are _lots_ of different collection traits (similar to C\#), but there doesn't
appear to be a "best" (unlike C\#, where `IEnumerable<T>` reigns).

Traversable is even more "general" than iterable. Deeper into the inheritance tree, there
is a dizzying array of all the standard stuff (maps, sets, lists, ...).

Scala has a _lot_ of methods for dealing with collections (more than C\# with LINQ).

Scala's `fold` and `reduce` seem similar to LINQ's aggregation.

Arithmetic operators (`:+`, `+=`, `++=`, ...) work for collections.

Beautiful syntax for creating dictionaries: `Map("a" -> 1, "b" -> 2)`, and doing arithmetic on maps.

It's important to understand the perforamnce of specific operations relative to
a specific collection type.

The `Vector` class is a good general-purpose immutable data structure, while `ArrayBuffer`
is a good "go-to" for mutable data structures.

The difference between "for / yield" and "map / flatMap"
[seems to involve monads](https://stackoverflow.com/a/14602182).

Make strings out of collections using `mkString` (better than C\#'s `string.Join`).

### Simple Build Tool (SBT)

The [Decompilers online](http://www.javadecompilers.com/) site may be useful for
examining the contents of artifacts.

Run `.\sbt.ps1 run` from the root of this repo to build and run the baby Scala
code under `src`. Run unit tests with `.\sbt.ps1 test` instead of `run`.

You can also install SBT [via Chocolatey](https://chocolatey.org/packages/sbt).

[ScalaTest](http://www.scalatest.org/) seems like a reasonably standard unit
testing framework.

Main commands: `compile`, `doc`, `package`, `run`, and `sbtVersion`.

Following [this StackOverflow answer](https://stackoverflow.com/a/43103983),
you can use `sbt 'test:testOnly *ClassName'` to only run tests in a specific
class.

Follow the [Scaladoc style guide](https://docs.scala-lang.org/style/scaladoc.html)
in writing docstrings.

You can list GitHub repositories as SBT project dependencies.

To create a single "fat JAR" (useful for deploying jobs to Spark clusters)
use the [assembly](https://github.com/sbt/sbt-assembly) plugin.

You can put general Scala logic in a `Build.scala` file (:warning: **this is now
deprecated**).

### Types

General point: Scala has a _very_ powerful type system.

Use `T[A+]`, `T[A-]` to constrain [type variance](https://stackoverflow.com/q/9619121).

Use `<:`, `:>` for type constraints.

In Scala, type parameters are typically called `A`, `B`, etc.

Structural typing (`A <: { def f() : Unit } `) is a proxy for duck typing.
:warning: _This uses reflection, so avoid when performance matters._

Immutable collections should take coveriant type parameters.

Implicit conversions are a way of telling the compiler "the obvious way
to make a `B` from an `A`."

Type classes seem like black magic.

Browse the Scala source code, e.g.
[Try.scala](https://github.com/scala/scala/blob/v2.9.3/src/library/scala/util/Try.scala),
to learn more about the type system.

### Idiomatic Scala

80\%/20\% rule: 80\% of your Scala code should be _pure_ (no side-effects)
and the other 20\% can connect your pure code with the outside world.

Prefer immutable objects whenever possible.

Use expressions (with no side effects) vs. statements (with hidden side effects).

Use match expressions, they're super powerful and flexible!

Don't _ever_ use `null` (use `Option` or `Try` instead).
