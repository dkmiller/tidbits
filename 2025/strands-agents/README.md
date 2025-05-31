# Strands agents SDK

https://strandsagents.com/0.1.x/user-guide/quickstart/

https://strandsagents.com/0.1.x/user-guide/concepts/model-providers/openai/

https://strandsagents.com/0.1.x/user-guide/deploy/deploy_to_aws_fargate/#containerization

https://strandsagents.com/0.1.x/user-guide/observability-evaluation/traces/

- [ ] [MCP](https://strandsagents.com/0.1.x/examples/python/mcp_calculator/)

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
fastapi dev demo_agent/api.py

# Old school
curl -H "content-type: application/json" localhost:8000/some_agent -d '{"prompt": "What is the 30th prime number divided by the 10th prime number."}'

# Modern & simpler, optionally with -v
http localhost:8000/some_agent prompt="What is the 30th prime number divided by the 10th prime number."
```

https://httpie.io/docs/cli/http-method

## Alternatives

https://www.reddit.com/r/AI_Agents/comments/1kjowzp/whats_the_best_framework_for_productiongrade_ai/

- https://google.github.io/adk-docs/
- https://microsoft.github.io/autogen/
