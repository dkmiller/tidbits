<#
Run all unit and integration tests.
#>

pip install -e $PSScriptRoot

pytest $PSScriptRoot
