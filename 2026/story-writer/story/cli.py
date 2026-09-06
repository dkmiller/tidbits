import logging
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter, validators

from story.ai import Ai

app = App()


@app.command
async def write(
    prompt: str,
    system: str = "system.md",
    setting: str = "setting.md",
    root: Annotated[
        Path, Parameter(validator=validators.Path(exists=True))
    ] = Path.cwd(),
    log_level: str = "INFO",
):
    logging.basicConfig(level=log_level)

    await Ai().novella((root / system).read_text(), setting, prompt)


app()
