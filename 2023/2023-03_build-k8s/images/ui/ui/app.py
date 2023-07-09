import streamlit as st
from find_primes import all_primes
from streamlit.logger import get_logger

# https://github.com/streamlit/streamlit/issues/4742
log = get_logger(__name__)


@st.cache_data
def prime_count(upper_bound: float) -> int:
    log.info("Calculating the number of primes below %s", upper_bound)
    primes = all_primes(upper_bound)
    return len(primes)


log.info("Page load")
st.title("Prime helper :abacus:!")


upper_bound = st.number_input("Upper bound", 2, value=int(1e8))
count = prime_count(upper_bound)


st.write(f"There are {count} primes <= {upper_bound}")


route = st.text_input("URL route", "/")
# impo

# TODO: call (https://www.tutorialworks.com/kubernetes-pod-communication/)
# http://api-service:8000
# if deployed and localhost if local.

# Make this a "proper" Python package which is installed in the Dockerfile.
# Ideally exposing "app" and "probe" commands which simplify the Dockerfile.


# def main():
#     from streamlit.web import bootstrap
#     # https://stackoverflow.com/a/76130057/
#     bootstrap.run(__file__, '', [], {})


# if __name__ == "__main__":
#     app()
