# SharePoint folder upload using Microsoft's Graph API

Title says it all.

Run example in PowerShell:

```powershell
dotnet run --configuration Release `
  --hostname microsoft.sharepoint.com `
  --team AMLXLIAristotleworkinggroup `
  --root \\FSU\Shares\TuringShare\ `
  --glob NLR_Models\Scripts\** `
  --num-fragments 30 `
  --target General `
  --token $tok `
```

## Links

- [Upload large files with an upload session](https://docs.microsoft.com/en-us/graph/api/driveitem-createuploadsession)
- [Upload a File to SharePoint using Azure Graph API](https://rahul-metangale.medium.com/upload-a-file-to-sharepoint-using-azure-graph-api-9deacce57449)
- [Microsoft Graph Explorer](https://developer.microsoft.com/en-us/graph/graph-explorer)
- [How to upload a large document in c\# using the Microsoft Graph API rest calls](https://stackoverflow.com/a/49780655)
- [FileUpload Task \#65](https://github.com/microsoftgraph/msgraph-sdk-dotnet-core/pull/65)
- [Capturing CancelKeyPress to stop an async console app at a safe point](https://stackoverflow.com/a/56372898)
- [How do I get a human-readable file size in bytes abbreviation using .NET?](https://stackoverflow.com/a/22366441)
- [C\# 8: Slicing with Indexes and Ranges](https://www.codejourney.net/2019/02/csharp-8-slicing-indexes-ranges/)
- [DefaultAzureCredential: Unifying How We Get Azure AD Token](https://www.rahulpnath.com/blog/defaultazurecredential-from-azure-sdk/)
- [Microsoft Graph API - OAuth 2.0 Scopes](https://docs.microsoft.com/en-us/answers/questions/28515/microsoft-graph-api-oauth-20-scopes.html)
- [Microsoft Graph permissions reference](https://docs.microsoft.com/en-us/graph/permissions-reference)
