from sentence_transformers import SentenceTransformer
import typer

main = typer.Typer()


@main.command()
def similarity(first: str = "I'm a meat crayon", second: str = "I'm a red crayon"):
    model = SentenceTransformer("BAAI/bge-large-zh-v1.5")
    embeddings_1 = model.encode([first], normalize_embeddings=True)
    embeddings_2 = model.encode([second], normalize_embeddings=True)
    similarity = embeddings_1 @ embeddings_2.T
    print(similarity)
