docker build -t proxy .

docker run --volume $PWD/work:/srv -p 8080:8080 proxy nginx -p /srv/ -c conf/nginx.conf
