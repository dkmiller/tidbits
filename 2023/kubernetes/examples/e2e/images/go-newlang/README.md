# Go newlang


https://go.dev/doc/tutorial/getting-started

To-do

- [ ] https://huma.rocks/tutorial/your-first-api/
- [ ] https://blog.devgenius.io/simple-rest-service-in-golang-with-openapi-spec-and-orm-a447b1086e21

## Running

```bash
go run . --config config.yaml

go mod tidy

docker build -t go-newlang . && docker run go-newlang --config foo

docker build -t go-newlang . && docker run --volume $PWD:/src2/ go-newlang --config /src2/config.yaml
```

## Links

- [vscode-remote-try-go](https://github.com/microsoft/vscode-remote-try-go/tree/main)
- [How to deploy a Go web application with Docker](https://semaphoreci.com/community/tutorials/how-to-deploy-a-go-web-application-with-docker)
- [Go by example: command-line flags](https://gobyexample.com/command-line-flags)
- [Parsing and generating YAML in Go](https://betterprogramming.pub/parsing-and-creating-yaml-in-go-crash-course-2ec10b7db850)
