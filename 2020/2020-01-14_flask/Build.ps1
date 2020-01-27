<#
Run Clean.ps1 (unless told not to) and then build the container specified
in this folder.
#>

param(
    [switch]$NoClean
)

if (!$NoClean) {
    .\Clean.ps1
}

docker build . -t danmill/flask
