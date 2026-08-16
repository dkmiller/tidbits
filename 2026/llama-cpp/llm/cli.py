import logging
from pathlib import Path

from cyclopts import App

from llm._jinja import JinjaRenderer
from llm._openai import ModelInvoker

log = logging.getLogger(__name__)


app = App()


# CLI to load from prompts/ directory containing .md files (either user
# messages or system prompts) and .yaml files (Jinja variables). The CLI format should
# look something like dan openai system_0.md free_1.md variables_0.yaml@0
# This loads markdown metadata to infer that system_0.md is a system prompt, free_1.md
# is a user message, and variables_0.yaml is a Jinja variable file. The @0 suffix
# indicates which index to use for the Jinja variables.
# Multiple prompt file _AND_ variable file arguments should be supported, and the CLI
# should be able to handle multiple system prompts and user messages in a single
# invocation. It will merge all the Jinja variables from the variable files and apply
# them to all the prompts


@app.command
async def generate(
    files: list[str],
    root: Path = Path.cwd(),
):
    """
    TODO mention variables (.yaml) and prompts (.md) files and how to use them
    together, also that order matters, and that you can use @0 to select an index of a
    variable file.
    """
    messages = JinjaRenderer().render_files(root, files)

    print(f"{messages=}")

    invoker = ModelInvoker(
        endpoint="http://dan-miller-dan-v8.ws.airdev.musta.ch:8080/v1",
        model="Qwen3.8-27b-uncensored",
        max_tokens=15000,
    )

    response = await invoker.invoke(messages)
    print(response)

    # TODO: basic "multi-turn" when files contains the special character
    # RESPOND.
