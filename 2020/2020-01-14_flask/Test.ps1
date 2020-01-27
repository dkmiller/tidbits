<#
Run Build.ps1 (unless told not to) and then call PyTest to ensure all the
local codebase works as expected.
#>

param(
    [switch]$NoBuild
)

$ErrorActionPreference = 'Stop'

if (!$NoBuild) {
    .\Build.ps1
}

pytest
