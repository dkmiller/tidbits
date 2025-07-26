# TODO: their docs are wrong (--port no longer works)
fastmcp run demo_agent/remote_tools.py --transport streamable-http &

fastapi dev demo_agent/api.py --port 8001
