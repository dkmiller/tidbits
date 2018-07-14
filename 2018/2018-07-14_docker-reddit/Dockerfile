# Super small Python image.
FROM stefanscherer/python-windows:nano

WORKDIR /app
COPY . /app

RUN pip install --trusted-host pypi.python.org -r requirements.txt
ENV NAME World
