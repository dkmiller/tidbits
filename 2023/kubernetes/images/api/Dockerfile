FROM python:3.11

WORKDIR /src/

COPY pyproject.toml /src/
COPY api/__init__.py /src/api/
RUN pip install -e .

COPY . /src/

# https://opentelemetry.io/docs/instrumentation/python/automatic/#configuring-the-agent
ENV OTEL_SERVICE_NAME=api
ENV OTEL_TRACES_EXPORTER=console,otlp
ENV OTEL_METRICS_EXPORTER=console,otlp
ENV OTEL_LOGS_EXPORTER=console
# Values from:
# https://github.com/grafana/helm-charts/tree/main/charts/tempo
ENV OTEL_EXPORTER_OTLP_ENDPOINT=http://grafana-tempo:4317

# Uncomment these to debug OpenTelemetry connectivity issues.
# ENV GRPC_VERBOSITY=debug
# ENV GRPC_TRACE=http,call_error,connectivity_state

# https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/fastapi/fastapi.html
ENV OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_SERVER_REQUEST=".*"
ENV OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_SERVER_RESPONSE=".*"

CMD [ "opentelemetry-instrument", "api"]
