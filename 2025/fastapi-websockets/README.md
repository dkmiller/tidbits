# FastAPI + reverse proxying websockets

https://wsh032.github.io/fastapi-proxy-lib/reference/fastapi_proxy_lib/core/websocket/#fastapi_proxy_lib.core.websocket.ReverseWebSocketProxy--examples

https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/realtime-audio-websockets

https://learn.microsoft.com/en-us/azure/ai-foundry/openai/realtime-audio-quickstart?tabs=keyless%2Cmacos&pivots=programming-language-python

"Pure" Python: https://platform.openai.com/docs/guides/realtime?connection-example=python#connect-with-websockets

How to handle secure / insecure websockets? Explicit `websocket_base_url`

``` bash
fastapi dev --port 11000
```

then

``` bash
python client.py
```

## Future

- [ ] WebRTC: https://dev.to/wassafshahzad/building-real-time-communication-harnessing-webrtc-with-fastapi-part-3-wrapping-every-thing-up-35fb
