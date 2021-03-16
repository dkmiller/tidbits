# Azure DevOps "identity" GUID

When "@mentioning" someone in an Azure DevOps wiki, the checked-in markdown file
in the underlying Git repository contains a mysterious GUID, e.g.:

> `@<8889F59B-406D-4FE6-A3BD-9DA142623119>`

This snippet provides code to find that ID, given a human-readable query like
"firstname lastname".

To run this, first `pip install` the declared dependencies, then run something
like:

```powershell
python get-ado-guid.py --org your_ado_organization --query "firstname lastname"
```

The script will print all "ADO IDs" (internally called `localId`) matching
that search. The underlying API call is **not** officially documented &mdash;
it was derived from "sniffing" REST calls using Edge's developer toolkit.

## Links

- [DevOps REST API: using the "userDescriptor" parameter](https://stackoverflow.com/a/63551756)
- [Could not find documentation about API IdentityPicker/Identities](https://developercommunity.visualstudio.com/t/could-not-find-documentation-about-api-identitypic/766560)
