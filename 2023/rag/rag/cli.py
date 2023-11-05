import typer

import rag

main = typer.Typer()


@main.command()
def similarity(first: str = "I'm a meat crayon", second: str = "I'm a red crayon"):
    rv = rag.embedding_similarity([first], [second])
    print(rv[0][0])
