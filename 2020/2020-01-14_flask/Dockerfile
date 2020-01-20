FROM continuumio/miniconda

RUN mkdir -p /app
WORKDIR  /app

# Speed up re-building the Docker image by initially copying over ONLY the
# dependencies (requirements) file.
COPY requirements.txt /app/
RUN conda install -y --file requirements.txt

ENV FLASK_ENV development
ENV FLASK_APP src/app.py

COPY . /app

EXPOSE 5000
ENTRYPOINT ["flask", "run", "--host=0.0.0.0"]
