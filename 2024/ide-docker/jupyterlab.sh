docker build -t jupyterlab -f jupyterlab.Dockerfile .

docker run -p 8080:8080 --volume $PWD:/src -w /src jupyterlab jupyter lab --allow-root --ip=0.0.0.0 --port=8080 --IdentityProvider.token=''
