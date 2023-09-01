# Mock OpenAI

Fake the Azure OpenAI REST API for help in unit testing. Start by learning how to proxy and test it.

## Running

This command will install dependencies and start a server on http://localhost:8000
which is API-compatible with Azure OpenAI.

```bash
pip install -e .

mock-openai
```

## Interactive


```python
import requests
import openai


requests.post("http://localhost:8000/config", json={"base": "https://*.openai.azure.com", "api_key": "..."})


openai.api_key = "sk-fake"
openai.api_base = "http://localhost:8000"
openai.api_type = "azure"
openai.api_version = "2023-07-01-preview"

response = openai.ChatCompletion.create(
    engine="gpt35turbo",
    messages=[{"role": "user", "content": "Write me a very long essay in the style of the US Declaration of Independence"}],
    stream=True,
)

for chunk in response:
    try:
        print(chunk["choices"][0]["delta"]["content"], end="", flush=True)
    except:
        pass

```

Directly interact with the streaming APIs.

```python

response = requests.post("http://localhost:8000/stream?size=10&sleep=0.8", stream=True)

for chunk in response.iter_lines():
    print(chunk)


response = requests.post("http://localhost:8000/stream_openai", stream=True)
for chunk in response.iter_lines():
    print(chunk)

```

## Profiling

Run a command like:

```bash
time curl 'http://localhost:8000/test-proxy?mode=aiohttp_singleton'
```

Better yet, install dev dependencies and use [Locust](https://locust.io/).

```bash
pip install -e ".[dev]"

locust
```
