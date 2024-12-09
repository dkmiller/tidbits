docker build -t jupyterlab -f jupyterlab.Dockerfile .

port=${1:-8080}

docker run \
  -p $port:$port \
  --volume $PWD:/src \
  -w /src \
  jupyterlab jupyter lab \
  --allow-root --ip=0.0.0.0 \
  --port=$port \
  --IdentityProvider.token=''
