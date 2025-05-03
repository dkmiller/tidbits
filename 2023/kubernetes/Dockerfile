FROM python:3.11

LABEL org.opencontainers.image.source=https://github.com/dkmiller/tidbits

WORKDIR /src/

COPY pyproject.toml /src/
COPY api/__init__.py /src/api/
RUN pip install -e .

COPY . /src/

# https://opentelemetry.io/docs/instrumentation/python/automatic/#configuring-the-agent
ENV OTEL_SERVICE_NAME=api
ENV OTEL_TRACES_EXPORTER=console
ENV OTEL_METRICS_EXPORTER=console
ENV OTEL_LOGS_EXPORTER=console

# Uncomment these to debug OpenTelemetry connectivity issues.
# ENV GRPC_VERBOSITY=debug
# ENV GRPC_TRACE=http,call_error,connectivity_state

# https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/fastapi/fastapi.html
ENV OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_SERVER_REQUEST=".*"
ENV OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_SERVER_RESPONSE=".*"

CMD [ "opentelemetry-instrument", "api"]
