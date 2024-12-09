docker build -t rstudio -f rstudio.Dockerfile .

# Bash argument with default: https://stackoverflow.com/a/9333006
port=${1:-8787}

# https://stackoverflow.com/a/47541663
# https://forum.posit.co/t/rstudio-server-behind-ingress-proxy-missing-cookie-info/134649
docker run \
  -e USER=rstudio \
  --platform=linux/amd64 \
  -p $port:$port \
  --volume $PWD:/home/rstudio/src \
  -w /src \
  rstudio \
  /usr/lib/rstudio-server/bin/rserver \
  --server-daemonize=0 \
  --auth-none 1 \
  --auth-validate-users 0 \
  --www-port $port \
  --server-user rstudio

# TODO: --www-root-path
