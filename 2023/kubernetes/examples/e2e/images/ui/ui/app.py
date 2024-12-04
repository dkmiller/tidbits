from urllib.parse import urljoin

import requests
import streamlit as st
from find_primes import all_primes
from opentelemetry import metrics, trace
from streamlit import runtime
from streamlit.logger import get_logger

from ui.builders import injector
from ui.models import Context

# https://github.com/streamlit/streamlit/issues/4742
log = get_logger(__name__)
meter = metrics.get_meter(__name__)
# https://www.vultr.com/docs/how-to-use-opentelemetry-with-streamlit-applications/
tracer = trace.get_tracer(__name__)


@st.cache_resource
def otel_counter(name: str, description: str):
    return meter.create_counter(name, description=description, unit="bytes")


@st.cache_data
@tracer.start_as_current_span("prime_count")
def prime_count(upper_bound: float) -> int:
    log.info("Calculating the number of primes below %s", upper_bound)
    primes = all_primes(upper_bound)
    return len(primes)


log.info("Page load")


st.set_page_config("UI", "🌐")
st.title("Prime helper :abacus:!")


upper_bound = st.number_input("Upper bound", 2, value=int(1e8))
count = prime_count(upper_bound)


st.write(f"There are {count} primes <= {upper_bound}")


route = st.text_input("URL route", "/httpbun")


context = injector().get(Context)

if context == context.DEPLOYED:
    log.info("Using service-to-service domain")
    # https://www.tutorialworks.com/kubernetes-pod-communication/
    domain = "api-service"
else:
    log.info("Using port-forwarded domain")
    domain = "localhost"


# https://stackoverflow.com/a/8223955
url = urljoin(f"http://{domain}:8000", route)


with st.spinner(f"Calling `{url}`"):
    with tracer.start_as_current_span("requests_get"):
        response = requests.get(url)
response.raise_for_status()

if trace_parent := response.json()["headers"].get("Traceparent"):
    trace_id = trace_parent.split("-")[1]
    st.write(f"Search in Grafana for `{trace_id}`")

st.write(response.json())


with tracer.start_as_current_span("streamlit.stats"):
    for stat in runtime.get_instance().stats_mgr.get_stats():
        # https://github.com/open-telemetry/opentelemetry-python/issues/1201
        otel_counter(stat.category_name, stat.cache_name).add(
            stat.byte_length,
        )
