# Docker on Windows

 from:

https://stefanscherer.github.io/is-there-a-windows-docker-image-for/


See https://github.com/docker/for-win/issues/1658

https://docs.docker.com/get-started/part2/#apppy

## Build and run

First, to build:

```cmd
docker build --tag friendlyhello .
```

Then, to run:

```cmd
docker run friendlyhello
```

Finally, `docker container ls` to verify that the container has stopped.
