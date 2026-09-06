## LLM-authored long-form stories (novellas etc.)

Installation and linting

``` bash
pip install -e "."
ruff format .
```

Flow.

``` mermaid
graph LR;

static_prompt --> setting

user_prompt --> setting

setting --> breakdown

static_breakdown_prompt --> breakdown

static_style_prompt

setting --> chapter

breakdown --> chapter

static_style_prompt --> chapter
```

## Reference

- https://github.com/dkmiller/modern-python-package/
- https://cyclopts.readthedocs.io/
- https://developers.openai.com/api/reference/python
