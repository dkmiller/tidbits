import random

import streamlit as st
from opentelemetry import metrics, trace
from streamlit import runtime

from otel import setup

meter = metrics.get_meter(__name__)
tracer = trace.get_tracer(__name__)


@st.cache_resource
def _counter(name: str, description: str):
    return meter.create_counter(name, description=description, unit="bytes")


@st.cache_data
@tracer.start_as_current_span("foo")
def foo(a):
    return a * 2


setup()


st.title("OpenTelemetry <> Streamlit")


with tracer.start_as_current_span("first_span"):
    st.write("Inside span")


st.write(f"Foo = `{foo(random.randint(0, 10000))}`")


stats = runtime.get_instance().stats_mgr

with tracer.start_as_current_span("streamlit.stats"):
    for stat in stats.get_stats():
        # https://github.com/open-telemetry/opentelemetry-python/issues/1201
        _counter(stat.category_name, stat.cache_name).add(
            stat.byte_length,
        )


trace.get_tracer_provider().force_flush()
