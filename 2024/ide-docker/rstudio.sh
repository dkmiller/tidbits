docker build -t rstudio -f rstudio.Dockerfile .

docker run --platform=linux/amd64 -p 8787:8787 rstudio /usr/lib/rstudio-server/bin/rserver --server-daemonize=0 --auth-none=1

# /usr/lib/rstudio-server/bin/rstudio --server-daemonize=0 --auth-none=1

# https://stackoverflow.com/a/47541663
# USER=rstudio /usr/lib/rstudio-server/bin/rserver --server-daemonize=0 --auth-none=1
