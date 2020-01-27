<#
Clean both the Docker context AND the local filesystem.

WARNING: this script is pretty brutal!
#>

$ErrorActionPreference = 'Stop'

docker ps -q | ForEach-Object { docker stop $_ }
docker system prune --force

git clean -xdf
