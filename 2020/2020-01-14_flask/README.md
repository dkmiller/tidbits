# Flask + PyTorch &mapsto; Azure

Run a small ML model in Azure via Flask, in a Docker container.

## Running and testing

Run by calling `.\Run.ps1`. Test by calling

```powershell
Invoke-WebRequest -Uri http://localhost:5000/predict -Method Post
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
