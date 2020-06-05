<#
TODO: ...
#>

param(
    $Subscription = '48bbc269-ce89-4f6f-9a12-c6f91fcb772d',
    $ResourceGroup = 'aml1p-rg',

    [switch]$SkipAadV2,
    [switch]$SkipAzVersion
)

if (!$SkipAzVersion) {
    az --version
}

az account set --subscription $Subscription

if (!$SkipAadV2) {
    # https://docs.microsoft.com/en-us/azure/aks/managed-aad
    az feature register --name AAD-V2 --namespace Microsoft.ContainerService
    az provider register -n Microsoft.ContainerService
}

az deployment group create `
    --name danmill-aks-learn `
    --resource-group $ResourceGroup `
    --template-file arm/template.json `
    --parameters @arm/parameters.json

