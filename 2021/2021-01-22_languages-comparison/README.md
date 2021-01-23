# Programming languages comparison

Lots of people say "you should use sexy new language X". Well, should you? This
document gives a number of **real-world** features that can be used to assess
the practicality of a language, and specific rules + examples for scoring
languages based on those features.

Maybe this list says more about the languages I've been exposed to than anything else ;).

In some ways, a counterpoint to
[_These Modern Programming Languages Will Make You Suffer_](https://medium.com/better-programming/modern-languages-suck-ad21cbc8a57c).
- Response: https://clocksmind.blogspot.com/2020/12/a-response-to-programming-language.html?fbclid=IwAR3HsYSsXVPPKB3waGLj9oNyRT4-GokpeK38ah-_D_IulN90VB7XplZB4Y8 .

## Features

For each of these features, "+1 point" for each bullet point.

### Dependency management

A real-world language has a dependency management framework, so that you can
"refer" to code that other people have written without just manually downloading
it all.

- Dependencies do not have to be installed globally (i.e., I can have two projects `A` and `B` on the same computer, which consume different versions of the dependency `D`).
- Configurable dependency source (i.e., the ability for packages to be resolved from multiple different "feeds").
- Authenticated feeds.
- Bonus points for a variety of version "masks" (e.g., `1.*`, `1.~`), along with "native" SemVer.

### Unit testing

Parameterized tests

Code coverage

### Docker

Should have standard, supported, base images.

### Serialization

CSV, JSON, Protobuf, XML, YAML.

### HTTP

Ability to make all the standard types of HTTP requests, with up-to-date encryption.

Generate client libs from Swagger endpoints. Useful even for dynamically typed languages.

### Encryption

Support for "standard" hashing functions, e.g. SHA-256.

Also, ability to load and use certificate files.

### Date times

Support for serialization, deserialization, time zones, etc.

### Regular expressions

'Nuff said.

### Reflection

Ability to interact with objects and types at runtime.
- Comes "for free" in most dynamically typed languages.

### Compilation

Ability to parse text into runnable code **at runtime**.

### Generics

Meaningful even for dynamically typed languages (I'm looking at you, Python) for good type hinting.

Bonus points for algebraic data types (make sure I know what that means).
- E.g., C\# doesn't have a good way of saying "this class is a Monad".

### IDE

Preferably a language server that integrates with e.g. Atom, Sublime, VS Code. Auto-completion, etc.

### Async

Some kind of threading, ideally more lightweight than "full" OS threads.

### Logging

Some kind of "opinionated" framework that's not just print statements.

Want to be able to configure levels, output to a file / online frameworks / command line.

## Examples

May want to rotate the table?

Also include:

- Erlang
- Kotlin
- Dart

| Feature | C\# | C++ | Elm | Haskell | Java | PureScript | Python |
|-|-|-|-|-|-|-|-|
| Dependency management | 3 | 0? | | | | | 3 |
