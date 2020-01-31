<#
Clean both the Docker context AND the local filesystem.

WARNING: this script is pretty brutal!
#>

$ErrorActionPreference = 'Stop'

Write-Warning 'Cleaning up Docker...'
docker ps -q | ForEach-Object { docker stop $_ }
docker system prune --force

Write-Warning 'Cleaning up local filesystem...'
git clean -xdf
