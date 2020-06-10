<#
Run both server (in a separate window) and client programs.
#>

param (
    [int]$Sleep = 10
)

Start-Process -WorkingDirectory $PSScriptRoot powershell {
    dotnet run --configuration Release --project Silo
}

# TODO: find a way to ping the silo until it's alive.
Write-Host "Sleeping $Sleep seconds..."
Start-Sleep $Sleep

dotnet run --configuration Release --project $PSScriptRoot/Client
