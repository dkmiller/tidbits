# Strands agents SDK

https://strandsagents.com/0.1.x/user-guide/quickstart/

https://strandsagents.com/0.1.x/user-guide/concepts/model-providers/openai/

https://strandsagents.com/0.1.x/user-guide/deploy/deploy_to_aws_fargate/#containerization

https://strandsagents.com/0.1.x/user-guide/observability-evaluation/traces/

[MCP](https://strandsagents.com/0.1.x/examples/python/mcp_calculator/)

- https://gofastmcp.com/deployment/cli#options
- Feature request: OTel in FastMCP: https://github.com/modelcontextprotocol/python-sdk/issues/421
- Feature request: OTel in MCP spec: https://github.com/modelcontextprotocol/modelcontextprotocol/issues/246
- https://gofastmcp.com/deployment/running-server

[Honeycomb &gt;  Send Data with the OpenTelemetry Collector](https://docs.honeycomb.io/send-data/opentelemetry/collector/)

- https://ui.honeycomb.io/dan-miller/environments/test/datasets/dan-strands-agents/home?tab=explore

``` mermaid
graph LR
caller -->|invokes| api

api["API"] -->|"wraps + invokes"| agent

mcp["MCP (Hosted tools)"]

agent -->|calls| mcp

agent -->|uses| llms["Hosted LLMs"]
```

## Running

``` bash
./serve.sh

# Old school
curl -H "content-type: application/json" localhost:8001/some_agent -d '{"prompt": "What is the 30th prime number divided by the 10th prime number."}'

# Modern & simpler, optionally with -v
http localhost:8001/primes_agent prompt="What is the 30th prime number divided by the 10th prime number."
```

https://httpie.io/docs/cli/http-method

## Future

- [ ] Structured agent API output
- [ ] Differential privacy and "sensitive" agent state
- [ ] Automatic conversion between OpenAPI &mapsto; FastMCP https://gofastmcp.com/servers/openapi

## Alternatives

https://www.reddit.com/r/AI_Agents/comments/1kjowzp/whats_the_best_framework_for_productiongrade_ai/

- https://google.github.io/adk-docs/get-started/quickstart/
    - OTel: https://github.com/google/adk-python/blob/main/src/google/adk/telemetry.py
- https://microsoft.github.io/autogen/
    - OTel: https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/framework/telemetry.html
- https://github.com/huggingface/smolagents
    - OTel: https://huggingface.co/docs/smolagents/en/tutorials/inspect_runs

Themes: agentic framework should:

- Produce OpenTelemetry-compatible traces
- Consume MCP servers
- Stretch: produce _both_ "sane" structured REST APIs and MCP servers
