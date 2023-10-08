# https://www.honeycomb.io/blog/open-telemetry-vendor-neutral
OTEL_SERVICE_NAME=fastapi-config \
  HONEYCOMB_API_KEY=$(op read op://Private/honeycomb/password) \
  OTEL_EXPORTER_OTLP_ENDPOINT=https://api.honeycomb.io \
  OTEL_EXPORTER_OTLP_HEADERS="x-honeycomb-team=$(op read op://Private/honeycomb/password)" \
  opentelemetry-instrument uvicorn main:app
# Sadly doesn't seem to work with --reload on.
