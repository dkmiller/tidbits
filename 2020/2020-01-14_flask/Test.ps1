<#
Run Build.ps1 (unless told not to) and then call PyTest to ensure all the
local codebase works as expected.
#>

param(
    [switch]$NoBuild,
    [switch]$Remote
)

$ErrorActionPreference = 'Stop'

if (!($NoBuild -or $Remote)) {
    .\Build.ps1
}

if ($Remote) {
    $Json = Get-Content arm/parameters.json | ConvertFrom-Json
    $Endpoint = "http://$($json.parameters.siteName.value).azurewebsites.net"
    $env:INFERENCE_ENDPOINT = $Endpoint
}

pytest

if ($Remote) {
    Remove-Item env:INFERENCE_ENDPOINT
}

if ($LASTEXITCODE -ne 0) {
    throw 'Tests failed!'
}
