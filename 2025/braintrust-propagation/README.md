# Braintrust + OTel context propagation

Compare context propagation between Braintrust's built-in tracing and
and their OpenTelemetry integration, in particular when conducting
Braintrust "experiments".

## Running

``` bash
pip install -r requirements.txt

export BRAINTRUST_API_KEY="..."
export BRAINTRUST_PROJECT_NAME="..."
export BRAINTRUST_DATASET_NAME="..."
export BRAINTRUST_API_URL="..."
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=$OTEL_EXPORTER_OTLP_TRACES_ENDPOINT/otel/v1/traces

python instrument_braintrust.py
```

## Links

https://opentelemetry.io/docs/concepts/signals/baggage/

https://opentelemetry-python.readthedocs.io/en/latest/api/baggage.html

https://opentelemetry-python.readthedocs.io/en/latest/api/baggage.propagation.html

https://www.braintrust.dev/docs/start/eval-sdk

https://www.braintrust.dev/docs/guides/experiments
