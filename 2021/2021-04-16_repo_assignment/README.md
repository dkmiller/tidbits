# Try programmatically responding to email

Sadly, Microsoft Graph doesn't get access :( .

Run the following command (copy/paste access token from
[Microsoft Graph Explorer](https://developer.microsoft.com/en-us/graph/graph-explorer)).

```powershell
python respond.py +access_token=$token
```

## Links

- https://docs.microsoft.com/en-us/graph/api/user-list-mailfolders
- https://docs.microsoft.com/en-us/graph/api/user-list-messages
- https://docs.microsoft.com/en-us/graph/query-parameters
- https://stackoverflow.com/a/33716764
- https://stackoverflow.com/a/5396320
