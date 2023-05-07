import streamlit as st


st.title("Streaming!")

placeholder = st.empty()


text = """
Hello there!
<break>
# Title

- Bullet
- Bullet
<break>

This is a sentance. <break> It does some stuff.

<break>

```rust
use std::error::Error;

async fn call_endpoint(url: &str) -> Result<String, Box<dyn Error>> { <break>
    let response = <break> reqwest::get(url).await?.text().await?;
    Ok(response)<break>
}
```
"""

s = "Hi\n"
for fragment in text.split("<break>"):
    with placeholder.container():
        s += fragment
        st.write(s)
        import time

        time.sleep(0.3)
