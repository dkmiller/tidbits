# Minimal APIs in .NET 6

Run the local server:

```powershell
dotnet run --project MinApi
```

Call the web API:

```powershell
Invoke-WebRequest http://localhost:5000/hello

iwr  "http://localhost:5000/user/$([guid]::NewGuid())"

# https://stackoverflow.com/a/35725547
iwr -me post http://localhost:5000/hello -Body '{"name": "dan miller", "age": 29}' -ContentType application/json

iwr http://localhost:5000/isadult?age=19
```

Format the code:

```
dotnet format MinApi
```

[Swagger UI](http://localhost:5000/swagger/).

## To-do

- [ ] Load record-typed configuration from app settings.
    - [Sadly, only landing in .NET 7](https://github.com/dotnet/runtime/issues/46299)
- [ ] Authorization (currently getting `No service for type 'Microsoft.AspNetCore.Authentication.IAuthenticationService' has been registered.`)

## Links

- [Building minimal APIs in .NET 6](https://dotnetcoretutorials.com/2021/07/16/building-minimal-apis-in-net-6/)
- [Minimal APIs at a glance](https://gist.github.com/davidfowl/ff1addd02d239d2d26f4648a06158727)
