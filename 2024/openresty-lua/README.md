# OpenResty + Lua

https://openresty.org/en/getting-started.html#start-the-nginx-server

https://github.com/luarocks/luarocks/

https://github.com/SkyLothar/lua-resty-jwt

## Run

First `./run.sh`, then any of the commands below.

```bash
curl localhost:8080

curl -H "Authorization: Bearer $(iap-auth)" localhost:8080

# TODO:  (https://cloud.google.com/iap/docs/signed-headers-howto)
```
