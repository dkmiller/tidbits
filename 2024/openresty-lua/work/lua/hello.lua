-- These require statements are cached: https://stackoverflow.com/a/34235619
local json = require "json"
local jwt = require "resty.jwt"

local _M = {}

function _M.greet(name)
    ngx.say("Greetings from ", name)
end

function _M.proxy_pass()
    -- https://github.com/openresty/lua-nginx-module#ngxreqget_headers
    local goog_iap_jwt_assertion = ngx.req.get_headers()["x-goog-iap-jwt-assertion"]
    local parsed_goog_iap_jwt_assertion = jwt:load_jwt(goog_iap_jwt_assertion)
    local access_levels = parsed_goog_iap_jwt_assertion["payload"]["google"]["access_levels"]

    -- For-loop: https://www.lua.org/pil/4.3.1.html
    for _, access_level in ipairs(access_levels) do
        if access_level:match("/accessLevels/hrl_network$") then
            ngx.var.lua_hrl = "true"
        end
    end

    ngx.var.lua_hello_header = json.encode(access_levels)
end

return _M
