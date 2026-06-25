import streamlit as st
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter


# https://stackoverflow.com/q/76076082
@st.cache_resource
def setup():
    st.write("setup")
    resource = Resource({"service.name": "otel-streamlit"})
    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(ConsoleSpanExporter())  # out=_Out()
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    # https://opentelemetry.io/docs/languages/python/exporters/
    reader = PeriodicExportingMetricReader(ConsoleMetricExporter())
    meterProvider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(meterProvider)
    # MeterProvider()
    # metrics.set


# https://github.com/open-telemetry/opentelemetry-python/discussions/3212
# https://github.com/open-telemetry/opentelemetry-python/issues/2802
