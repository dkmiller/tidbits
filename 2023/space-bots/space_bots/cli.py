import typer

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def hello(name: str):
    print(f"Hello {name}")
