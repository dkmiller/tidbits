FROM --platform=linux/amd64 ubuntu:24.04

RUN apt --fix-broken install

RUN apt-get update && apt-get install -y \
  dirmngr \
  gdebi-core \
  software-properties-common \
  wget

RUN wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee -a /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc \
  && add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $(lsb_release -cs)-cran40/"

# RUN apt install --no-install-recommends r-base
RUN apt-get update && apt-get install -y r-base

# https://unix.stackexchange.com/a/596250
RUN wget https://download2.rstudio.org/server/jammy/amd64/rstudio-server-2024.09.1-394-amd64.deb \
  && gdebi -n rstudio-server-2024.09.1-394-amd64.deb

# RUN wget https://download1.rstudio.org/electron/jammy/amd64/rstudio-2024.09.1-394-amd64.deb && apt install -y ./rstudio-2024.09.1-394-amd64.deb
