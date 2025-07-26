import logging
from functools import lru_cache

from cache_decorator import Cache
from sentence_transformers import SentenceTransformer

# https://stackoverflow.com/a/33045252/
# Embeddable = str | list[str]


log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def embedding_model():
    return SentenceTransformer("BAAI/bge-large-zh-v1.5")


@Cache()
def embedding(content: list[str]):
    log.info("Calculating embeddings for %s chunks", len(content))
    model = embedding_model()
    return model.encode(content, normalize_embeddings=True)


@Cache()
def embedding_similarity(left: list[str], right: list[str]):
    embeddings_1 = embedding(left)
    embeddings_2 = embedding(right)
    similarity = embeddings_1 @ embeddings_2.T
    return similarity
