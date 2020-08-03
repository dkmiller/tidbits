FROM conda/miniconda3

COPY environment.yml /tmp/

RUN conda update -n base -c defaults conda
RUN conda env create --file /tmp/environment.yml

RUN useradd --create-home app
WORKDIR /home/app
USER app

COPY . .

ENTRYPOINT [ "bash", "/home/app/run.sh" ]
