import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import FastAPI, Query, Request

from azure.identity import ClientSecretCredential

log = logging.getLogger("uvicorn")

app = FastAPI()


EXPIRES_IN = 3599
FORMAT_STRING = "%m/%d/%Y %H:%M:%S +00:00"


@app.get("/")
# No path is used
# https://stackoverflow.com/a/64147450
# @app.get("/{full_path:path}")
# The only call appears to be GET.
# @app.post("/{full_path:path}")
async def identity(
    request: Request,
    api_version: Annotated[str, Query(alias="api-version")],
    resource: str,
):
    method = request.method
    headers = request.headers
    body = await request.body()
    url = request.url

    log.info(
        """Request info:
Method: %s
API Version: %s
Resource: %s
URL: %s
Headers:
    %s
Body:
    %s
    """,
        method,
        api_version,
        resource,
        url,
        headers,
        body,
    )

    now = datetime.now(timezone.utc)
    expiry = now + timedelta(seconds=EXPIRES_IN)
    expires_on = expiry.strftime(FORMAT_STRING)

    rv = {
        "access_token": "fake_access_token",
        "refresh_token": "",
        "expires_in": EXPIRES_IN,
        "expires_on": expires_on,
        "not_before": expires_on,
        "resource": resource,
        "token_type": "Bearer",
    }

    log.info("Response: %s", rv)
    return rv


@app.get("/passthrough")
async def passthrough(resource: str):
    # TODO: should we just re-use DefaultAzureCredential here? This makes things
    # explicit.
    credential = ClientSecretCredential(
        os.environ["AZURE_TENANT_ID"],
        os.environ["AZURE_CLIENT_ID"],
        os.environ["AZURE_CLIENT_SECRET"],
    )

    log.info("Resource: %s", resource)

    # TODO: this needs to convert "resources" to "scopes".
    # Strange, even if the client-side Python code requests */.default, the Azure Identity SDK
    # strips /.default from it...
    if not resource.endswith("/.default"):
        resource += "/.default"
    token = credential.get_token(resource)

    now = datetime.now()
    # https://stackoverflow.com/a/3682808
    expiry = datetime.utcfromtimestamp(token.expires_on)
    expires_on = expiry.strftime(FORMAT_STRING)

    rv = {
        "access_token": token.token,
        "expires_in": (expiry - now).seconds,
        "expires_on": expires_on,
        "not_before": expires_on,
        "resource": resource,
        "token_type": "Bearer",
    }

    return rv
