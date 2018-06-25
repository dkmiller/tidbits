# Docker on Windows

Playing around with using Docker on Windows, following their
[Get Started](https://docs.docker.com/get-started/) guide. See
[Is there a Windows Docker image for ...?](https://stefanscherer.github.io/is-there-a-windows-docker-image-for/)
for a list of useful Docker images, e.g. ones that solve
[this issue](See https://github.com/docker/for-win/issues/1658).

## Build and run

Follow the script below:

```powershell
# Build
docker build --tag friendlyhello .

# Run
docker run friendlyhello
```

Finally, `docker container ls` to verify that the container has stopped.
