<#
Run Test.ps1 (unless told not to) and then deploy the created resources to
Azure.

The overall sequence is:

Clean → Build → Test → Deploy
#>

param(
    [switch]$NoDocker,
    [switch]$NoTest,
    $Subscription = '1247cb5d-803a-4ca0-9fb5-0b1e23a002d2',
    $ResourceGroup = 'dockerflasktutorial',
    $Version = '0.0.0'
)

$ErrorActionPreference = 'Stop'

if (!$NoTest) {
    .\Test.ps1
}

Write-Host 'Setting Azure subscription...'
az account set --subscription $Subscription

$Json = Get-Content arm/parameters.json | ConvertFrom-Json
$Location = $Json.parameters.location.value

Write-Host 'Creating resource group...'
az group create --name $ResourceGroup --location $Location

Write-Host 'Deploying Azure resources...'
az group deployment create `
  --name 'Docker_Flask_Example' `
  --resource-group $ResourceGroup `
  --template-file arm/template.json `
  --parameters @arm/parameters.json

if (!$NoDocker) {
    $AcrName = $Json.parameters.registryName.value

    Write-Host 'Logging into Azure Container Registry (ACR)...'
    az acr login --name $AcrName

    Write-Host 'Deploying container to ACR...'
    $Image = "$($AcrName).azurecr.io/danmill/flask:$Version"
    docker tag danmill/flask $Image
    docker push $Image

    Write-Host 'Obtaining username and password for ACR...'
    $Credential = az acr credential show --name $AcrName | ConvertFrom-Json

    Write-Host 'Setting web app to use deployed container...'
    az webapp config container set `
      --name $Json.parameters.siteName.value `
      --resource-group $ResourceGroup `
      --docker-custom-image-name $Image `
      --docker-registry-server-url https://$AcrName.azurecr.io `
      --docker-registry-server-user $Credential.username `
      --docker-registry-server-password $Credential.passwords[0].value

      Write-Host 'Waiting for web app to initialize...'
      Start-Sleep -Seconds 60
      .\Test.ps1 -Remote
}
