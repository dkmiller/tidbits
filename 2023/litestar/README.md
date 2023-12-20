# Litestar

https://docs.litestar.dev/2/

```bash
pip install -e ".[dev]"

uvicorn api.main:app --reload

# Lint
ruff format . && ruff --fix . && isort --profile black .
```

## Thoughts

- It's annoying to need to explicitly pass in all routes to the `Litestar`
  constructor.
- The built-in command `litestar run` is too inflexible. It does not support
  configurable entry file names or directories.
    - Its [autodiscovery](https://docs.litestar.dev/latest/usage/cli.html#autodiscovery)
      is pretty good though, but it assumes things are in a folder named `app`.
    - :x: Their docs suggest `litestar run --app api.main:app`, which sadly
      does not work.
- `/schema` (ReDoc) is less functional than Swagger: it does not appear to
  support making HTTP requests.
- `/schema/elements` doesn't provide any meaningful additional functionality
  beyond Swagger.
- `/schema/rapidoc` is nice (shows response headers) but not necessarily worth
  the transition.
- Emphasis on Pydantic + controllers might pay off with an especially large
  or complex API surface.
- Litestar requires explicit return type annotations, it cannot infer them
  (unlike FastAPI).
- It looks like [Litestar _should_ support DataDog via their ASGI
  integration](https://github.com/litestar-org/litestar/issues/2402)
- [Official OpenTelemetry integration](https://docs.litestar.dev/latest/usage/metrics/open-telemetry.html#opentelemetry)
    - Sadly, configuration via the command line does not appear to work,
      [unlike with FastAPI](https://github.com/dkmiller/tidbits/blob/main/2023/kubernetes/examples/e2e/images/api/Dockerfile)
      ```bash
      OTEL_TRACES_EXPORTER=console OTEL_SERVICE_NAME=api opentelemetry-instrument uvicorn api.main:app --reload
      ```
- Litestar warns on non-`async` routes, this is nice.

## Conclusion

Litestar appears to be a fast, flexible, broadly used ASGI framework. It
requires slightly more boilerplate than FastAPI for small projects, but it
has a few more batteries included (ORM, auth) for large and complex ones.
It has a much smaller community
([78k GitHub results](https://www.google.com/search?q=site%3Agithub.com+%22litestar%22)
vs.
[FastAPI's 2.8m](https://www.google.com/search?q=site%3Agithub.com+%22fastapi%22))
and no "killer feature".

Unless there is some specific feature of Litestar that is make or break (e.g.,
specific doc page etc.) I would recommend going with FastAPI.
