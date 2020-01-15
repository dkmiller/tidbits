# Flask + PyTorch &mapsto; Azure

Run a small ML model in Azure via Flask, in a Docker container.

## Links

- [Deploying PyTorch in Python via a REST API with Flask](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html)
- [Speed Up The Build of Your Python Docker Image](https://vsupalov.com/speed-up-python-docker-image-build/)
- [pytest: helps you write better programs](https://docs.pytest.org/en/latest/)
- [Changing standard (Python) test discovery](http://doc.pytest.org/en/latest/example/pythoncollection.html)

## Running and testing

Run by calling `.\Run.ps1`. Test by calling

```powershell
Invoke-WebRequest -Uri http://127.0.0.1:5000/predict -Method Post
```
