def test_chat_completion_nostream(openai):
    chat_completion = openai.ChatCompletion.create(
        engine="gpt35turbo", messages=[{"role": "user", "content": "Hello world"}]
    )
    assert chat_completion


def test_chat_completion_streaming(openai):
    chat_completion = openai.ChatCompletion.create(
        engine="gpt35turbo",
        messages=[
            {
                "role": "user",
                "content": "Write me a very long essay in the style of the US Declaration of Independence",
            }
        ],
        stream=True,
    )

    deltas = []
    errors = []
    for x in chat_completion:
        try:
            deltas.append(x["choices"][0]["delta"]["content"])
        except:
            errors.append(x)

    assert len(deltas) > 20
    assert len(errors) <= 2
