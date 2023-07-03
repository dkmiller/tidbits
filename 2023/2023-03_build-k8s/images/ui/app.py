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
