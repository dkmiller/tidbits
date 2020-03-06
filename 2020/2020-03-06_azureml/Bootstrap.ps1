<#

#>

param (
    $Email = 'danmill@microsoft.com',
    $Name = 'Dan Miller'
)

$File = 'dsdevops-oh-files.zip'
$Path = Join-Path $PSScriptRoot $File
$Url = "https://mlopsohdata.blob.core.windows.net/mlopsohdata/$File"

Invoke-WebRequest -Uri $Url -OutFile $Path

Expand-Archive $Path -Force

git config --global user.email $Email
git config --global user.name $Name
