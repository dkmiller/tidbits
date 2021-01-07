# SharePoint folder upload using Microsoft's Graph API

Title says it all.

Run example in PowerShell:

```powershell
dotnet run --configuration Release `
  --access-token $tok -`
  -hostname microsoft.sharepoint.com `
  --team AMLXLIAristotleworkinggroup `
  --file-path \\FSU\Shares\TuringShare\ `
  --file-pattern NLR_Models\Multilingual\TULRv1\** `
  --target-path General
```

## Links

- [Upload large files with an upload session](https://docs.microsoft.com/en-us/graph/api/driveitem-createuploadsession)
- [Upload a File to SharePoint using Azure Graph API](https://rahul-metangale.medium.com/upload-a-file-to-sharepoint-using-azure-graph-api-9deacce57449)
- [Microsoft Graph Explorer](https://developer.microsoft.com/en-us/graph/graph-explorer)
- [How to upload a large document in c\# using the Microsoft Graph API rest calls](https://stackoverflow.com/a/49780655)
- [FileUpload Task \#65](https://github.com/microsoftgraph/msgraph-sdk-dotnet-core/pull/65)
- [Capturing CancelKeyPress to stop an async console app at a safe point](https://stackoverflow.com/a/56372898)
- [How do I get a human-readable file size in bytes abbreviation using .NET?](https://stackoverflow.com/a/22366441)
