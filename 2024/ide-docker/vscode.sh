docker build -t vscode -f vscode.Dockerfile .

docker run -p 8080:8080 --volume $PWD:/src -w /src vscode code-server --bind-addr 0.0.0.0:8080 --auth none
