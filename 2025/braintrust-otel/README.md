# Braintrust + OTel

Experiment with OTel-style traces propagated to Braintrust.

``` bash
pip install -r requirements.txt

fastapi dev

curl -H "content-type: application/json" localhost:8000/chat -d '{"name": "dan"}'

ruff check --fix . && ruff format .
```

- https://www.braintrust.dev/docs/guides/traces/integrations#opentelemetry-otel
- https://fastapi.tiangolo.com/advanced/events/#lifespan-function
- https://opentelemetry.io/docs/languages/python/instrumentation/
- https://opentelemetry.io/docs/languages/python/exporters/#usage
- https://www.braintrust.dev/app/dkmiller/p/dkmiller-first-project/
- https://github.com/openai/openai-python?tab=readme-ov-file#async-usage

TODO

- [ ] `OTEL_PYTHON_EXCLUDED_URLS`
