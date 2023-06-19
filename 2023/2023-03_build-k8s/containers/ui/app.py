from find_primes import all_primes
import streamlit as st


@st.cache_data
def prime_count(upper_bound: float) -> int:
    primes = all_primes(upper_bound)
    return len(primes)


st.title("Prime helper :abacus:!")


upper_bound = st.number_input("Upper bound", 2, value=int(1e8))
count = prime_count(upper_bound)


st.write(f"There are {count} primes <= {upper_bound}")
