FROM ubuntu:22.04

ARG LUA_VERSION=5.3.5
ARG LUAROCKS_VERSION=3.11.1

# https://openresty.org/en/linux-packages.html#ubuntu

RUN apt-get update \
  && apt-get -y install \
  build-essential \
  ca-certificates \
  lsb-release \
  gnupg \
  libreadline-dev \
  unzip \
  wget

RUN wget -O - https://openresty.org/package/pubkey.gpg | gpg --dearmor -o /usr/share/keyrings/openresty.gpg

RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/openresty.gpg] http://openresty.org/package/arm64/ubuntu $(lsb_release -sc) main" | tee /etc/apt/sources.list.d/openresty.list > /dev/null

RUN apt-get update \
  && apt-get -y install openresty

ENV PATH=/usr/local/openresty/nginx/sbin:$PATH

# https://github.com/luarocks/luarocks/wiki/Installation-instructions-for-Unix#1ubuntudebian-user
RUN curl -R -O https://www.lua.org/ftp/lua-${LUA_VERSION}.tar.gz \
  && tar -zxf lua-${LUA_VERSION}.tar.gz \
  && cd lua-${LUA_VERSION} \
  && make linux test \
  && make install

RUN curl -R -O https://luarocks.github.io/luarocks/releases/luarocks-${LUAROCKS_VERSION}.tar.gz \
    && tar -zxf luarocks-${LUAROCKS_VERSION}.tar.gz \
    && cd luarocks-${LUAROCKS_VERSION} \
    && ./configure --with-lua-include=/usr/local/include \
    && make \
    && make install

# https://github.com/SkyLothar/lua-resty-jwt?tab=readme-ov-file#installation-1
RUN luarocks install lua-resty-jwt
