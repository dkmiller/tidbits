import logging

import base64
import asyncio
from openai import AsyncAzureOpenAI


logging.basicConfig(level="DEBUG")


AZURE_ENDPOINT = "https://eastus2.api.cognitive.microsoft.com/"
WEBSOCKET_BASE = "wss://eastus2.api.cognitive.microsoft.com/openai"

AZURE_ENDPOINT = "http://localhost:11000/experimental/"
WEBSOCKET_BASE = "ws://localhost:11000/experimental/openai"



async def main() -> None:
    """
    When prompted for user input, type a message and hit enter to send it to the model.
    Enter "q" to quit the conversation.
    """

    # websocket_base_url
    client = AsyncAzureOpenAI(
        # TODO: is the first one even necessary?
        azure_endpoint=AZURE_ENDPOINT,
        # It adds /realtime.
        websocket_base_url=WEBSOCKET_BASE,
        azure_ad_token="placeholder",
        # Placeholder API keys specified here overwrite the "real" proxy-side
        # API keys and cause HTTP 403s.
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
