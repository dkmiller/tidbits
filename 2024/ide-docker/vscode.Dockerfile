FROM ubuntu:24.04

RUN apt-get update && apt-get install -y curl

RUN curl -fsSL https://code-server.dev/install.sh | sh
