# Flask + PyTorch &mapsto; Azure

Run a small ML model in Azure via Flask, in a Docker container.

## Running and testing

Run by calling `.\Run.ps1`. Test by calling

```powershell
# TODO: why can't you test /predict this way?
Invoke-WebRequest -Uri http://localhost:5000/
```

Test by calling `pytest`.

## Links

- [Deploying PyTorch in Python via a REST API with Flask](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html)
- [Speed Up The Build of Your Python Docker Image](https://vsupalov.com/speed-up-python-docker-image-build/)
- [pytest: helps you write better programs](https://docs.pytest.org/en/latest/)
- [Changing standard (Python) test discovery](http://doc.pytest.org/en/latest/example/pythoncollection.html)
- [How to fix "Attempted relative import in non-package" even with `__init__.py`](https://stackoverflow.com/q/11536764)
- [`@pytest.mark.parametrize`: parametrizing test functions](https://docs.pytest.org/en/latest/parametrize.html)
- [Docker + Flask | A Simple Tutorial](https://medium.com/@doedotdev/docker-flask-a-simple-tutorial-bbcb2f4110b5)
- [Best practices for writing Dockerfiles](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)
- [Dockerize a Flask App](https://dev.to/riverfount/dockerize-a-flask-app-17ag)
- [Cross-platform way of getting temp directory in Python](https://stackoverflow.com/a/847866)
- [Download Files with Python](https://stackabuse.com/download-files-with-python/)
- [Getting started quickly with Flask logging](https://www.scalyr.com/blog/getting-started-quickly-with-flask-logging/)
- [How to use `Invoke-RestMethod` to upload jpg](https://stackoverflow.com/q/42395638)
- [classic xunit-style setup](https://docs.pytest.org/en/latest/xunit_setup.html)
- [How to start a background process in Python?](https://stackoverflow.com/a/7224186)
- [Killing a process created with Python's `subprocess.Popen()`](https://stackoverflow.com/q/4084322)
- [PEP 263 -- Defining Python Source Code Encodings](https://www.python.org/dev/peps/pep-0263/)
- [Stop all docker containers at once on Windows](https://stackoverflow.com/a/48813850)
- [Building Minimal Docker Containers for Python Applications](https://blog.realkinetic.com/building-minimal-docker-containers-for-python-applications-37d0272c52f3)
