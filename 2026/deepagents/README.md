# DeepAgents exploration

[![LangSmith](https://img.shields.io/badge/langsmith-tidbits--deepagents-blue?logo=opentelemetry)](https://smith.langchain.com/o/b35779dd-4399-451e-a8bd-c1ece5413807/projects/p/9b795a23-60c3-4b1a-8da0-3426dfec2c95)

Setup and running.

``` bash
pip install -e "."

fastapi dev

ruff format . && ruff check --fix .
```

then...

``` bash
curl localhost:8000/healthz

curl -H 'Content-Type: application/json' localhost:8000/invoke -d '{"messages": [{"role": "user", "content": "Plan and then write a short science fiction story with a surprising and dark twist at the end. Write it to a file STORY.md"}]}' | jq -r '.files["/STORY.md"].content'
```


## Backlog

- [ ] Tool giving [GraphQL-based access](https://docs.github.com/en/graphql/guides/forming-calls-with-graphql)
  to my personal GitHub
- [ ] Tests or [evals](https://docs.langchain.com/langsmith/evaluation-concepts#evaluators)
