FROM python:3.13-slim

RUN pip install fastapi[standard]==0.115.5

COPY . /srv/

WORKDIR /srv/

CMD ["fastapi", "dev", "main.py", "--host", "0.0.0.0"]
