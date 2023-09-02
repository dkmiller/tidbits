FROM python:3.11

WORKDIR /src/

COPY pyproject.toml /src/
COPY api/__init__.py /src/api/
RUN pip install -e .

COPY . /src/

CMD ["api"]
