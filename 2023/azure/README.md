# Azure managed identity

https://blog.jongallant.com/2021/08/azure-identity-202/


https://learn.microsoft.com/en-us/azure/app-service/overview-managed-identity

https://aka.ms/azsdk/python/identity/managedidentitycredential/troubleshoot

## Sample

URL: http://127.0.0.1:8000/?api-version=2017-09-01&resource=https://management.azure.com/

Headers:

({'host': '127.0.0.1:8000', 'user-agent': 'azsdk-python-identity/1.5.0 Python/3.9.6 (macOS-13.1-arm64-arm-64bit)', 'accept-encoding': 'gzip, deflate, br', 'accept': '*/*', 'connection': 'keep-alive', 'secret': 'msi_secret_value'})

Body:

```python
b''
```
