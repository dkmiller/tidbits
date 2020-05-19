FROM python:3.8-slim-buster

# Speed up re-building the Docker image by initially copying over ONLY the
# dependencies (requirements) file.
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

RUN useradd --create-home app
WORKDIR /home/app
USER app

COPY . .

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "app.py"]
