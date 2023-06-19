FROM python:3.11

COPY . /src/
WORKDIR /src/

RUN pip install -r requirements.txt

CMD ["uvicorn", "main:app"]
