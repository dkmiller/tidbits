# FastAPI + reverse proxying websockets

Demonstrate the combination of the following components.

- [fastapi-proxy-lib : reverse web socket proxy](https://wsh032.github.io/fastapi-proxy-lib/reference/fastapi_proxy_lib/core/websocket/#fastapi_proxy_lib.core.websocket.ReverseWebSocketProxy--examples)
- [How to use the GPT-4o Realtime API via WebSockets (Preview)](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/realtime-audio-websockets)
- [GPT-4o Realtime API for speech and audio (Preview)](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/realtime-audio-quickstart?tabs=keyless%2Cmacos&pivots=programming-language-python)

The key trick is to handle insecure websocket connections by explicitly
specifying the `websocket_base_url` constructor parameter for the OpenAI
client object.

## Running

``` bash
fastapi dev --port 11000
```

then

``` bash
python client.py

curl localhost:11000/experimental/openai
```

## Future

- [ ] WebRTC: https://dev.to/wassafshahzad/building-real-time-communication-harnessing-webrtc-with-fastapi-part-3-wrapping-every-thing-up-35fb
