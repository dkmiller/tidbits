import pandas as pd
import streamlit as st


path = st.text_input("Data", value="embeddings.parquet")

df = pd.read_parquet(path).dropna()

st.write(df.sample(15))


# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

import numpy as np
# https://stackoverflow.com/a/46913929/
X = np.asarray(df["embedding"].tolist(), dtype=np.float32)
model = pca.fit(X)


from skl2onnx import to_onnx

onx = to_onnx(model, X[:1])
with open("embeddings_project.onnx", "wb") as f:
    f.write(onx.SerializeToString())


# https://onnx.ai/sklearn-onnx/


# import onnxruntime as rt

# sess = rt.InferenceSession("embeddings_project.onnx", providers=["CPUExecutionProvider"])
# input_name = sess.get_inputs()[0].name
# label_name = sess.get_outputs()[0].name
# pred_onx = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]

Y = model.transform(X)

df[["x", "y"]] = Y



import plotly.express as px


# https://docs.streamlit.io/library/api-reference/charts/st.plotly_chart
fig = px.scatter(
    df,
    x="x",
    y="y",
    # size="pop",
    # color="glob",
    hover_name="text",
    # log_x=True,
    # size_max=60,
)


st.plotly_chart(fig, theme="streamlit", use_container_width=True)
