FROM python:3.13-slim

WORKDIR /src

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . /src

ENTRYPOINT ["fastapi", "dev", "server", "--host", "0.0.0.0"]
