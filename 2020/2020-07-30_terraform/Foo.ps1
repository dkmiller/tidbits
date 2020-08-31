param(
    $ClientId = '',
    [Parameter(Mandatory=$true)]$Secret,
    $Subscription = '48bbc269-ce89-4f6f-9a12-c6f91fcb772d',
    $Tenant = ''
)

Write-Host 'Hi!'

az login --service-principal -u <service_principal_name> -p "<service_principal_password>" --tenant "<service_principal_tenant>"

$Current = $PWD

try {
    Set-Location $PSScriptRoot
    terraform init
    terraform plan


    # az login --subscription $SubscriptionId
    # az account set --subscription $SubscriptionId

    # Set-Location $PSScriptRoot
    # docker run --volume ${PWD}:/src -w /src hashicorp/terraform:light apply
} finally {
    Set-Location $Current
}
