FROM ubuntu:22.04

# https://openresty.org/en/linux-packages.html#ubuntu

RUN apt-get update \
  && apt-get -y install wget gnupg ca-certificates lsb-release

RUN wget -O - https://openresty.org/package/pubkey.gpg | gpg --dearmor -o /usr/share/keyrings/openresty.gpg

RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/openresty.gpg] http://openresty.org/package/arm64/ubuntu $(lsb_release -sc) main" | tee /etc/apt/sources.list.d/openresty.list > /dev/null

RUN apt-get update \
  && apt-get -y install openresty

ENV PATH=/usr/local/openresty/nginx/sbin:$PATH
