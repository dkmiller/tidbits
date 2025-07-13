import logging

import base64
import asyncio
from dotenv import dotenv_values
from openai import AsyncAzureOpenAI


logging.basicConfig(level="DEBUG")


async def main() -> None:
    """
    When prompted for user input, type a message and hit enter to send it to the model.
    Enter "q" to quit the conversation.
    """

    # websocket_base_url
    client = AsyncAzureOpenAI(
        # TODO: is the first one even necessary?
        azure_endpoint="https://eastus2.api.cognitive.microsoft.com/",#"ws://localhost:8880/experimental",
        # websocket_base_url="wss://eastus2.api.cognitive.microsoft.com/",#"ws://localhost:8880/experimental",
        api_key=dotenv_values()["api_key"],
        api_version="2025-04-01-preview",
    )
    async with client.beta.realtime.connect(
        model="gpt-4o-realtime-preview",
    ) as connection:
        await connection.session.update(session={"modalities": ["text", "audio"]})  
        user_input = "hi there!"

        await connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": user_input}],
            }
        )
        await connection.response.create()
        async for event in connection:
            if event.type == "response.text.delta":
                print(event.delta, flush=True, end="")
            elif event.type == "response.audio.delta":

                audio_data = base64.b64decode(event.delta)
                print(f"Received {len(audio_data)} bytes of audio data.")
            elif event.type == "response.audio_transcript.delta":
                print(f"Received text delta: {event.delta}")
            elif event.type == "response.text.done":
                print()
            elif event.type == "response.done":
                break

asyncio.run(main())
