from pathlib import Path
import streamlit as st

import rag


st.title("RAG!")


st.write("Hi")

from importlib import reload
import rag.chunking
import rag.data
reload(rag.chunking)
reload(rag.data)

from rag.data import load_raw
from rag.chunking import chunkify, chunks


conf = rag.Config()

raw = load_raw("../../../notes/", "**/*.md")
f"Raw data has {len(raw)} paths"

chunked = chunkify(raw, conf)
f"Found **{len(chunked)}** chunks"

st.write(chunked)

import pandas as pd


# JSONDecodeError: Extra data: line 20 column 2 (char 1234)
def _foo(c):
    try:
        return rag.embedding(c)
    except Exception as e:
        st.warning(f"`{e}` `{c}`")
        return pd.NA


with st.spinner(f"Calculating **{len(chunked)}** embeddings"):
    _emb = chunked["chunk"].apply(_foo)




reload(pd)




chunked["embedding"] = _emb
chunked.to_parquet("embeddings.parquet", engine="pyarrow")

st.stop()


new_rows = []
import pandas as pd


for index, row in raw.iterrows():
    if index >= 5:
        break
    _c = chunks(row["text"], conf)
    for c_i, c in enumerate(_c):
        new_rows.append(pd.concat([row, pd.Series([c, c_i], index=["chunk", "chunk_index"])]))


st.write(pd.DataFrame(data=new_rows))

st.stop()

import pandas as pd
def f(group):
    row = group.irow(0)

    return pd.DataFrame({'chunk': chunks(row["text"], conf)})


df = raw.groupby("path", group_keys=False).apply(f)

st.write(df)

# st.write(
#     raw.apply(lambda row: chunks(row["text"], config), axis=1, result_type="expand")


# st.write(chunk(raw, conf))

st.stop()



root = st.text_input("Root", "../../../notes/")

_defaults = {
    0: "airbnb/**/*.md",
    1: "microsoft/**/*.md",
}

n_globs = st.number_input("N globs", min_value=1, value=2)
globs = []
for index in range(n_globs):
    globs.append(st.text_input(f"Glob {index+1}", _defaults.get(index)))



# glob1 = st.text_input("Glob", "airbnb/**/*.md")
# glob2 = st.text_input("Glob", "microsoft/**/*.md")
limit = st.number_input("Limit", min_value=1, value=10)

# paths = []

# for glob in globs:



texts = []
vectors = []
_globs = []
_paths = []

for index, glob in enumerate(globs):
    paths = list(Path(root).glob(glob))
    paths = sorted(paths, key=lambda p: p.stem)[:limit]
    st.write(f"Found {len(paths)} paths for `{glob}`")
    # paths.append(glob_paths[:limit])

    for path in paths:
        chunks = rag.chunks(path.read_text(), conf)
        texts.extend(chunks)
        embeddings = rag.embedding(chunks)
        vectors.extend(embeddings)
        _globs.extend([glob] * len(chunks))
        _paths.extend([str(path.absolute().resolve())] * len(chunks))


# st.write(_globs)



# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

import numpy as np
# https://stackoverflow.com/a/46913929/
npa = np.asarray(vectors, dtype=np.float32)
npa_t = pca.fit_transform(npa)

import pandas as pd

df = pd.DataFrame(data=npa_t, columns=["x", "y"])

df["glob"] = _globs
df["path"] = _paths
df["text"] = texts

st.write(df)


import plotly.express as px


# https://docs.streamlit.io/library/api-reference/charts/st.plotly_chart
fig = px.scatter(
    df,
    x="x",
    y="y",
    # size="pop",
    color="glob",
    hover_name="text",
    # log_x=True,
    # size_max=60,
)


st.plotly_chart(fig, theme="streamlit", use_container_width=True)
