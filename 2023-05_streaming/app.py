import streamlit as st


st.title("Streaming!")

placeholder = st.empty()

s = "Hi "
for index in range(200):
    with placeholder.container():
        s += f" {index} "
        st.write(s)
        import time
        time.sleep(.1)
