import pandas as pd
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

from rag.config import Config


def chunks(text: str, config: Config) -> list[str]:
    # https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/code_splitter
    md_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN, chunk_size=config.char_limit, chunk_overlap=0
    )
    chunks = md_splitter.create_documents([text])
    return [chunk.page_content for chunk in chunks]


def chunkify(raw: pd.DataFrame, config: Config):
    """
    New columns are `chunk` (raw text) and `chunk_index` (index).
    """
    rows = []
    for _, row in raw.iterrows():
        row_chunks = chunks(row["text"], config)
        for index, text in enumerate(row_chunks):
            # Concat: https://stackoverflow.com/a/76104173/
            rows.append(pd.concat([row, pd.Series([text, index], index=["chunk", "chunk_index"])]))
    return pd.DataFrame(data=rows)
