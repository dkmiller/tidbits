<#
Run Test.ps1 (unless told not to) and then deploy the created resources to
Azure.

The overall sequence is:

Clean → Build → Test → Deploy
#>

param(
    [switch]$NoTest,
    $Subscription = '1247cb5d-803a-4ca0-9fb5-0b1e23a002d2',
    $Location = 'westus2',
    $RgName = 'dockerflasktutorial',
    $AcrName = 'flaskacr01',
    $AcrSku = 'Basic'
)

$ErrorActionPreference = 'Stop'

if (!$NoTest) {
    .\Test.ps1
}

az account set --subscription $Subscription

az deployment create `
    --location $Location `
    --template-file arm/template.json `
    --parameters rgName=$RgName rgLocation=$Location acrName=$AcrName acrSku=$AcrSku
