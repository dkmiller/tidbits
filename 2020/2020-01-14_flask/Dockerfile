# Avoid the alpine images; they have obscure bugs due to non-standard gcc
# installation.
FROM python:3.8-slim-buster

# Speed up re-building the Docker image by initially copying over ONLY the
# dependencies (requirements) file.
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

RUN useradd --create-home app
WORKDIR /home/app
USER app

COPY . .

ENV FLASK_ENV development
ENV FLASK_APP src/app.py
EXPOSE 5000
ENTRYPOINT ["flask", "run", "--host=0.0.0.0"]
