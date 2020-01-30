<#
Run Test.ps1 (unless told not to) and then deploy the created resources to
Azure.

The overall sequence is:

Clean → Build → Test → Deploy
#>

param(
    [switch]$NoTest,
    $Subscription = '1247cb5d-803a-4ca0-9fb5-0b1e23a002d2',
    $ResourceGroup = 'dockerflasktutorial'
)

$ErrorActionPreference = 'Stop'

if (!$NoTest) {
    .\Test.ps1
}

az account set --subscription $Subscription

$Json = Get-Content arm/parameters.json | ConvertFrom-Json
$Location = $Json.parameters.location.value

az group create --name $ResourceGroup --location $Location

az group deployment create `
  --name 'Docker_Flask_Example' `
  --resource-group $ResourceGroup `
  --template-file arm/template.json `
  --parameters @arm/parameters.json

  # $Image = "$($AcrName).azurecr.io/danmill/flask:$Version"

# az deployment create `
#     --location $Location `
#     --template-file arm/template.json `
#     --parameters rgName=$RgName rgLocation=$Location acrName=$AcrName acrSku=$AcrSku dockerTag=$Image

# az acr login --name $AcrName

# docker tag danmill/flask $Image
# docker push $Image
