docker ps -q | ForEach-Object { docker stop $_ }
docker system prune --force
git clean -xdf
