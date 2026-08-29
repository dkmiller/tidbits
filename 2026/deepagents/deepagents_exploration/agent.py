from deepagents import create_deep_agent


SYSTEM_PROMPT = """You are an expert researcher. Your job is to conduct thorough research and then write a polished report.

You have access to an internet search tool as your primary means of gathering information.

## `internet_search`

Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.
"""


# TODO: dependency injection.
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-6",
    # tools=[internet_search],
    system_prompt=SYSTEM_PROMPT,
)
