# Strangely, this does not seem to work with python:3.8-alpine.
FROM continuumio/miniconda3

RUN mkdir -p /app
WORKDIR  /app

# Latest versions of PyTorch (especially torchvision) are hosted on the pytorch
# channel.
RUN conda config --append channels pytorch

# PyTest is hosted in this channel.
RUN conda config --append channels conda-forge

# Speed up re-building the Docker image by initially copying over ONLY the
# dependencies (requirements) file.
COPY requirements.txt /app/
RUN conda install -y --file requirements.txt

ENV FLASK_ENV development
ENV FLASK_APP src/app.py

COPY . /app

EXPOSE 5000
ENTRYPOINT ["flask", "run", "--host=0.0.0.0"]
