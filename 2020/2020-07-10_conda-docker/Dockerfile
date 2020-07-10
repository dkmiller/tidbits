FROM continuumio/miniconda3

WORKDIR /src
COPY environment.yml .
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "data-science", "/bin/bash", "-c"]

COPY run.py .
ENTRYPOINT ["conda", "run", "-n", "data-science", "python", "run.py"]
