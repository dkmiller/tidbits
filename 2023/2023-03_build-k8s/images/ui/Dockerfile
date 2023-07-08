# https://stackoverflow.com/a/71611002
# TODO: --platform=linux/amd64
FROM python:3.11

# TODO: bring this back.
# https://github.com/davidteather/TikTok-Api/issues/95#issuecomment-673789306
# https://linux.how2shout.com/install-chromium-browser-debian-11-bullseye/
# https://stackoverflow.com/a/71935536
# RUN apt-get update && apt-get install -y chromium chromium-driver

WORKDIR /src/
COPY requirements.txt /src/

RUN pip install -r requirements.txt

COPY . /src/

CMD ["streamlit", "run", "app.py"]
