from urllib.parse import urljoin

import requests
import streamlit as st
from find_primes import all_primes
from streamlit.logger import get_logger
from ui.builders import injector
from ui.models import Context

# https://github.com/streamlit/streamlit/issues/4742
log = get_logger(__name__)


@st.cache_data
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


route = st.text_input("URL route", "/")


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
    response = requests.get(url)
response.raise_for_status()

st.write(response.json())
