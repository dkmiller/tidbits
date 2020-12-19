FROM node

RUN npm install -g gulp-cli

COPY . /src
WORKDIR  /src

RUN npm install
RUN gulp
