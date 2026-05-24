# DeepAgents exploration

Setup and running.

``` bash
pip install -e "."

export ANTHROPIC_API_KEY="..."
export LANGSMITH_API_KEY=...


fastapi dev
```

then...

``` bash
curl localhost:8000/healthz

curl -H 'Content-Type: application/json' localhost:8000/invoke -d '{"messages": [{"role": "user", "content": "Plan and then write a short science fiction story with a surprising and dark twist at the end"}]}'
```

https://smith.langchain.com/o/b35779dd-4399-451e-a8bd-c1ece5413807/projects/p/9b795a23-60c3-4b1a-8da0-3426dfec2c95

## Backlog

- [ ] https://github.com/1Password/onepassword-sdk-python/
- [ ] Tool giving [GraphQL-based access](https://docs.github.com/en/graphql/guides/forming-calls-with-graphql)
  to my personal GitHub
