# Docker on Windows

Playing around with using Docker on Windows, following their
[Get Started](https://docs.docker.com/get-started/) guide. See
[Is there a Windows Docker image for ...?](https://stefanscherer.github.io/is-there-a-windows-docker-image-for/)
for a list of useful Docker images, e.g. ones that solve
[this issue](https://github.com/docker/for-win/issues/1658).

**_Goal:_** find a way (using Docker) to run .NET Core (or a specific
.NET Framework version), Python, and Gradle all in parallel.

See [this question](https://stackoverflow.com/a/24958548/) for an
explanation of `ADD` vs. `COPY` in Docker.

## Build and run

The script [Run.ps1](./Run.ps1) builds and runs Docker images using
both Python and .NET. It's a lightweight proxy for a Makefile which
could call such code.
