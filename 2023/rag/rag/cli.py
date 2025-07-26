from pathlib import Path
import typer

import rag

# https://stackoverflow.com/a/76375308/
main = typer.Typer(pretty_exceptions_enable=False)


@main.command()
def similarity(first: str = "I'm a meat crayon", second: str = "I'm a red crayon"):
    rv = rag.embedding_similarity([first], [second])
    print(rv[0][0])


@main.command()
def split(root: str = "../../../notes/", glob: str = "**/*.md", limit: int = 5):

    paths = list(Path(root).glob(glob))
    paths = sorted(paths, key=lambda p: p.stem)
    print(f"Found {len(paths)} paths: {paths[:limit]}")

    conf = rag.Config()

    vectors = []

    for path in paths[:limit]:
        print(f"Splitting {path}")
        chunks = rag.chunks(path.read_text(), conf)
        print(f"Got {len(chunks)} chunks")
        for chunk in chunks:
            print(f"\t{chunk}")
            print("-" * 70)
        print("=" * 80)
        embeddings = rag.embedding(chunks)
        vectors.extend(embeddings)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)

    import numpy as np
    # https://stackoverflow.com/a/46913929/
    npa = np.asarray(vectors, dtype=np.float32)
    down = pca.fit_transform(npa)
    print(down)


@main.command()
def ui():
    pass
    from streamlit.web import bootstrap

    script = (Path(__file__).parent / "ui" / "app.py").absolute()
    bootstrap.run(str(script), "", [], {})
