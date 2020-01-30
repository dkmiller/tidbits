# Flask + PyTorch &mapsto; Azure

Run a small ML model in Azure via Flask, in a Docker container.

## Running and testing

Run by calling `.\Run.ps1`. Test by calling

```powershell
# Sadly, you can't use iwr to test the /predict endpoint, because
# it doesn't encode things the same way as Python does.
Invoke-WebRequest -Uri http://localhost:5000/

# After deploy:
Invoke-WebRequest -Uri https://danmill-inference.azurewebsites.net/
```

Test by calling `pytest`.

After deploying, navigate to:
- http://danmill-inference.azurewebsites.net/
- https://danmill-inference.scm.azurewebsites.net/api/logs/docker

## Troubleshooting

If the running web app has issues you can restart it via

```powershell
az webapp restart --name danmill-inference --resource-group dockerflasktutorial
```

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
- [New Pillow version (7.0.0) breaks torchvision (ImportError: cannot import name 'PILLOW_VERSION' from 'PIL')](https://github.com/pytorch/vision/issues/1712)
- [luiscape/install_packages.sh](https://gist.github.com/luiscape/19d2d73a8c7b59411a2fb73a697f5ed4#gistcomment-2879010)
- [Logging within `py.test` tests](https://stackoverflow.com/q/4673373)
- [Docker SDK for Python](https://docker-py.readthedocs.io/en/stable/)
- [Anaconda Terminal error `pywin32_bootstrap`](https://stackoverflow.com/a/59194389)
- [How to stop a PowerShell script on the first error?](https://stackoverflow.com/q/9948517)
- [Deploy resources with Resource Manager templates and Azure CLI](https://docs.microsoft.com/en-us/azure/azure-resource-manager/templates/deploy-cli)
- [Create resource groups and resources at the subscription level](https://docs.microsoft.com/en-us/azure/azure-resource-manager/templates/deploy-to-subscription)
- [azure-quickstart-templates](https://github.com/Azure/azure-quickstart-templates/blob/master/101-container-registry/azuredeploy.json)
- [Push your first image to a private Docker container registry using the Docker CLI](https://docs.microsoft.com/en-us/azure/container-registry/container-registry-get-started-docker-cli)
- [Tutorial: Build a custom image and run in App Service from a private registry](https://docs.microsoft.com/en-us/azure/app-service/containers/tutorial-custom-docker-image)
- [Production-ready Docker packaging](https://pythonspeed.com/docker/)
- [Azure Resource Manager template functions](https://docs.microsoft.com/en-us/azure/azure-resource-manager/templates/template-functions)
- https://stackoverflow.com/a/4907053
- https://stackoverflow.com/a/58370117
- https://blogsprajeesh.blogspot.com/2015/02/powershell-get-set-and-remove.html
