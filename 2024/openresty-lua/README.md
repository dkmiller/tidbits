# OpenResty + Lua

https://openresty.org/en/getting-started.html#start-the-nginx-server

https://github.com/luarocks/luarocks/

https://github.com/SkyLothar/lua-resty-jwt

https://github.com/craigmj/json4lua

https://jwt.io/

https://blog.openresty.com/en/or-lua-module/

## Run

First `./run.sh`, then any of the commands below.

```bash
curl localhost:8080

curl -H "Authorization: Bearer $(iap-auth)" localhost:8080

# TODO:  (https://cloud.google.com/iap/docs/signed-headers-howto)
```

## Lua links

- List of keys in a table https://stackoverflow.com/a/12674376
- Concatenate list of strings https://stackoverflow.com/a/59118638
- String interpolation http://lua-users.org/wiki/StringInterpolation
- String split https://stackoverflow.com/a/2780182
