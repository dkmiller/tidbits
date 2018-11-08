<#
TODO: documentation.
#>

$Current = $PWD

try {
    Copy-Item $PSScriptRoot/docker-compose.yml docker-spark
    Set-Location $PSScriptRoot/docker-spark
    docker-compose up
} catch {
    Write-Host 'Something went wrong!'
} finally {
    Set-Location $Current
}


