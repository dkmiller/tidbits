# FastAPI + OpenTelemetry + hot reload

They don't "naively" play together via `opentelemetry-instrument`. Can we make
something work anyways?

Inspired by:
https://github.com/dkmiller/tidbits/blob/67fecb2af5388dfa6961900fc439645175b36042/2023/kubernetes/examples/e2e/images/api/api/main.py

```bash
localhost:8000/probe
```

## Links

- https://opentelemetry.io/docs/languages/python/instrumentation/#acquire-tracer
