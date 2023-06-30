import typer
from rich import print

app = typer.Typer()


@app.command()
def build(src: str = ".", output: str = ".build"):
    print(f"Building [bold]{src}[/bold] ↦ [bold]{output}[/bold]")


@app.command()
def deploy(name: str, formal: bool = False):
    if formal:
        print(f"Goodbye Ms. {name}. Have a good day.")
    else:
        print(f"Bye {name}!")
