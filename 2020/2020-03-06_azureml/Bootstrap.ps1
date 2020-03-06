<#
Run this once after cloning the main repo.
#>

param (
    $Email = 'danmill@microsoft.com',
    $Name = 'Dan Miller'
)

$File = 'dsdevops-oh-files.zip'
$Path = Join-Path $PSScriptRoot $File
$Url = "https://mlopsohdata.blob.core.windows.net/mlopsohdata/$File"

if (!(Test-Path $Path)) {
    Invoke-WebRequest -Uri $Url -OutFile $Path
}

if (!(Test-Path 'dsdevops-oh-files')) {
    Expand-Archive $Path -Force
}

$Script = Join-Path $PSScriptRoot 'upload.py'
python $Script

git config --global user.email $Email
git config --global user.name $Name
git config --global push.default simple
