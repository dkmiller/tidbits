FROM python:3.11

WORKDIR /src/
COPY requirements.txt /src/

RUN pip install -r requirements.txt

COPY . /src/

ENV adf987=a

# https://stackoverflow.com/a/72001015
ENV STREAMLIT_SERVER_HEADLESS=true
CMD ["streamlit", "run", "app.py"]
