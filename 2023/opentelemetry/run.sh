OTEL_RESOURCE_ATTRIBUTES=service.name=toph opentelemetry-instrument \
    --traces_exporter console \
    --logs_exporter console \
   --metrics_exporter console \
    uvicorn toph.app:app 
    # https://signoz.io/blog/opentelemetry-fastapi/
    # --reload
