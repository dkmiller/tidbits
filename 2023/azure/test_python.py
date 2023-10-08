import os
from unittest import mock

import pytest
import requests

PATCHED_ENV = {
    "IDENTITY_HEADER": "identity_header_value",
    "IDENTITY_ENDPOINT": "bogus",
    "MSI_ENDPOINT": "http://127.0.0.1:8000",
    "MSI_SECRET": "msi_secret_value",
}


@mock.patch.dict(os.environ, PATCHED_ENV)
def test_managed_identity_credential():
    from azure.identity import ManagedIdentityCredential

    credential = ManagedIdentityCredential()
    token = credential.get_token("https://management.azure.com/")
    assert token
    assert token.token == "fake_access_token"


@pytest.mark.parametrize("resource", ["https://management.azure.com/"])
# https://adamj.eu/tech/2020/10/13/how-to-mock-environment-variables-with-pytest/
@mock.patch.dict(os.environ, PATCHED_ENV)
def test_default_azure_credential(resource):
    from azure.identity import DefaultAzureCredential

    credential = DefaultAzureCredential()
    # ManagedIdentityCredential: This credential requires exactly one scope per token request.
    # https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.managedidentitycredential?view=azure-python
    token = credential.get_token(resource)
    assert token
    assert token.token == "fake_access_token"


@pytest.mark.parametrize(
    "resource",
    ["https://management.azure.com/", "https://management.azure.com/.default"],
)
@mock.patch.dict(
    os.environ, {**PATCHED_ENV, "MSI_ENDPOINT": "http://127.0.0.1:8000/passthrough"}
)
def test_get_subscriptions(resource):
    from azure.identity import DefaultAzureCredential

    credential = DefaultAzureCredential()
    token = credential.get_token(resource)
    assert token

    response = requests.get(
        "https://management.azure.com/subscriptions?api-version=2022-12-01",
        headers={"Authorization": f"Bearer {token.token}"},
    )
    response.raise_for_status()
    assert response.json()["value"]


@mock.patch.dict(
    os.environ, {**PATCHED_ENV, "MSI_ENDPOINT": "http://127.0.0.1:8000/passthrough"}
)
def test_get_secret():
    from azure.identity import DefaultAzureCredential

    credential = DefaultAzureCredential()
    token = credential.get_token("https://vault.azure.net/.default")
    assert token

    response = requests.get(
        "https://puppy-monitor-kv.vault.azure.net/secrets?api-version=7.4",
        headers={"Authorization": f"Bearer {token.token}"},
    )
    response.raise_for_status()
    assert response.json()["value"]
