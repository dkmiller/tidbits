# Azure managed identity

Simple REST API for understanding and eventually replacing the Azure "managed identity" API that
their client SDK calls to obtain bearer tokens.

Helpful links:

- [Azure Identity 202 - Environment Variables](https://blog.jongallant.com/2021/08/azure-identity-202/)
- [How to use managed identities for App Service and Azure Functions](https://learn.microsoft.com/en-us/azure/app-service/overview-managed-identity)
- [Troubleshoot Azure Identity authentication issues](https://aka.ms/azsdk/python/identity/managedidentitycredential/troubleshoot)

## Sample

> GET http://127.0.0.1:8000/?api-version=2017-09-01&resource=https://management.azure.com/

Headers:

```json
{
    "host": "127.0.0.1:8000",
    "user-agent": "azsdk-python-identity/1.5.0 Python/3.9.6 (macOS-13.1-arm64-arm-64bit)",
    "accept-encoding": "gzip, deflate, br",
    "accept": "*/*",
    "connection": "keep-alive",
    "secret": "msi_secret_value"
}
```

Empty body.
