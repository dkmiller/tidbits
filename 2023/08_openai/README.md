# Mock OpenAI

Fake the REST API for help in unit testing.

## Running

This command will install dependencies and start a server on http://localhost:8000
which is API-compatible with Azure OpenAI.

```bash
pip install -e .

mock-openai

# uvicorn mock_openai.app:app --reload
```

## Interactive

```python
import requests


response = requests.post("https://air-openai-test2.openai.azure.com/openai/deployments/gpt35turbo/chat/completions?api-version=2023-03-15-preview", headers={"api-key": "..."}, json={"messages": [{"role": "system", "content": "You are an assistant"}, {"role": "user", "content": "Who is Dan Miller?"}]})


# Write me a lengthy essay in the style of the US Declaration of Independence

response = requests.post("https://air-openai-test2.openai.azure.com/openai/deployments/gpt35turbo/chat/completions?api-version=2023-07-01-preview", headers={"api-key": "..."}, json={"messages": [{"role": "system", "content": "You are an assistant"}, {"role": "user", "content": "Count from one to one hundred, with the new number on a new line each time."}], "stream": True, "top_p": .95, "temperature": .7, "max_tokens": 1000}, stream=True)

for line in response.iter_lines():
    print(line)

```


...

```python
import openai

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

...

```python
import requests

response = requests.post("http://localhost:8000/stream?size=10&sleep=0.8", stream=True)

for chunk in response.iter_lines():
    print(chunk)



response = requests.post("http://localhost:8000/stream_openai", stream=True)
for chunk in response.iter_lines():
    print(chunk)

```




```python
response = requests.post("https://air-openai-test2.openai.azure.com/openai/deployments/gpt35turbo/chat/completions?api-version=2023-07-01-preview", headers={"api-key": "..."}, json={"messages": [{"role": "system", "content": "You are an assistant"}, {"role": "user", "content": "Count from one to one hundred, with the new number on a new line each time."}], "stream": True}, stream=True)
for line in response.iter_lines():
    print(line)
```