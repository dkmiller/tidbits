FROM continuumio/miniconda

RUN mkdir -p /app
WORKDIR  /app
# Speed up re-building the Docker image by initially copying over ONLY the
# dependencies (requirements) file.
COPY requirements.txt /app/requirements.txt
RUN conda install -y --file requirements.txt

COPY . /app
