OTEL_RESOURCE_ATTRIBUTES=service.name=toph opentelemetry-instrument \
  --traces_exporter console \
  --logs_exporter console \
  --metrics_exporter console \
  uvicorn toph.app:app
  # Auto instrumentation doesn't work with reload enabled.
  # https://signoz.io/blog/opentelemetry-fastapi/
  # --reload
