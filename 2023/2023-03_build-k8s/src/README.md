# Ptah

Named after the
[Egyptian god of craftsmen and architects](https://en.wikipedia.org/wiki/Ptah).

[Installing from remote source](https://stackoverflow.com/a/19516714):

```bash
pip install "git+https://github.com/dkmiller/tidbits.git#egg=ptah&subdirectory=2023/2023-03_build-k8s/src"
```

## To-do

- [ ] Handle port forwarding for all relevant deployments + the "admin" UI:
  ```bash
  kubectl port-forward deployment/ui-deployment 8501:8501 &
  kubectl proxy
  ```

## Links

- https://typer.tiangolo.com/
- https://injector.readthedocs.io/
- https://cachelib.readthedocs.io/file/
- https://rich.readthedocs.io/
- https://docker-py.readthedocs.io/
