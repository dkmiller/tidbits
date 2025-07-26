docker build -t vscode -f vscode.Dockerfile .

port=${1:-8080}

docker run \
  -p $port:$port \
  --volume $PWD:/src \
  -w /src \
  vscode code-server \
  --bind-addr 0.0.0.0:$port \
  --auth none
