FROM r-base

COPY packages.r /src/
RUN Rscript /src/packages.r

COPY src /src/
WORKDIR /src/
