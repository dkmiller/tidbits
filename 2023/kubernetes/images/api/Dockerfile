FROM python:3.11

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

CMD [ "opentelemetry-instrument", "api"]
