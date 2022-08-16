FROM python:3.10

COPY requirements.txt /reqs/
RUN pip install -r /reqs/requirements.txt

COPY src /src/
WORKDIR /src/
