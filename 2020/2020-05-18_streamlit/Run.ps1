docker build -t streamlit-app $PSScriptRoot

docker run -p 8501:8501 streamlit-app
