# Braintrust + OTel context propagation

Compare context propagation between Braintrust's built-in tracing and
and their OpenTelemetry integration, in particular when conducting
Braintrust "experiments".

https://airbnb.slack.com/archives/C08N2C6B077/p1752779784058039

https://opentelemetry.io/docs/concepts/signals/baggage/

https://opentelemetry-python.readthedocs.io/en/latest/api/baggage.html

https://opentelemetry-python.readthedocs.io/en/latest/api/baggage.propagation.html

https://www.braintrust.dev/docs/start/eval-sdk

https://www.braintrust.dev/docs/guides/experiments

## Running

``` bash
pip install -r requirements.txt

export export BRAINTRUST_API_KEY=sk-...
```
