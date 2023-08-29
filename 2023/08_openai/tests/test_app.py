import openai


def test_chat_completion_nostream():
    # https://github.com/openai/openai-python#microsoft-azure-endpoints
    openai.api_key = "sk-fake"
    openai.api_base = "http://localhost:8000"
    openai.api_type = "azure"
    openai.api_version = "2023-03-15-preview"

    chat_completion = openai.ChatCompletion.create(
        engine="gpt35turbo", messages=[{"role": "user", "content": "Hello world"}]
    )
    assert chat_completion


def test_chat_completion_streaming():
    # https://github.com/openai/openai-python#microsoft-azure-endpoints
    openai.api_key = "sk-fake"
    openai.api_base = "http://localhost:8000"
    openai.api_type = "azure"
    openai.api_version = "2023-03-15-preview"

    chat_completion = openai.ChatCompletion.create(
        engine="gpt4",
        messages=[{"role": "user", "content": "Write me a very long essay in the style of the US Declaration of Independence"}],
        stream=True,
    )

    deltas = []
    bad = []
    for x in chat_completion:
        delta = x["choices"][0]["delta"]
        if "content" in delta:
            deltas.append(delta["content"])
        else:
            bad.append(x)
            # raise Exception(delta)
        # deltas.append(x["choices"][0]["delta"]["content"])
    assert chat_completion
    # if bad:
    #     raise Exception(bad)
    raise Exception("".join(deltas))


# def test_raw_requests():
#     import requests
#     response = requests.post("https://air-openai-test2.openai.azure.com/openai/deployments/gpt35turbo/chat/completions?api-version=2023-03-15-preview", headers={"api-key": "..."}, json={"messages": [{"role": "system", "content": "You are an assistant"}, {"role": "user", "content": "Who is Dan Miller?"}]})
