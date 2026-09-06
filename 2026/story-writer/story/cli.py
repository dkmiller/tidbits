import logging
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter, validators

from story.ai import Ai

app = App()


@app.command
async def write(
    root: Annotated[
        Path, Parameter(validator=validators.Path(exists=True))
    ] = Path.cwd(),
    chapters: int = 10,
    model: str = "grok-4.6-nomoderation",
    log_level: str = "INFO",
):
    logging.basicConfig(level=log_level)

    for i in range(chapters):
        print(f"Looping in {root}! {i}")

    await Ai(model=model).hello()


app()
