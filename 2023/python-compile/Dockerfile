FROM python:3.10

WORKDIR /src/

COPY pyproject.toml /src/
COPY pyc/__init__.py /src/pyc/
RUN pip install -e .

COPY . /src/

# https://stackoverflow.com/a/60875372/
ENTRYPOINT [ "pyinstaller", "/src/pyc/cli.py", "--onefile", "--clean", "--windowed" ]
