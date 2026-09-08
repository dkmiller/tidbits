import logging
from pathlib import Path
from typing import Annotated

from cyclopts import App, Parameter, validators

from story.ai import Ai

CWD = Path.cwd()

app = App()


@app.command
async def write(
    prompt: str,
    system: str = "system.md",
    setting: str = "setting.md",
    breakdown: str = "breakdown.md",
    style: str = "style.md",
    root: Annotated[
        Path, Parameter(validator=validators.Path(exists=True, file_okay=False))
    ] = CWD,
    log_level: str = "INFO",
):
    logging.basicConfig(level=log_level)

    novella = await Ai().novella(
        system=(root / system).read_text(),
        setting=(root / setting).read_text(),
        prompt=prompt,
        breakdown=(root / breakdown).read_text(),
        style=(root / style).read_text(),
    )

    print("-" * 80)
    print(novella)


app()
