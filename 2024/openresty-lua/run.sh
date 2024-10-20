docker build -t proxy .

docker run --volume $PWD/work:/srv -p 8080:8080 proxy /bin/bash -c "nginx -p /srv/ -c conf/nginx.conf; sleep infinity"
