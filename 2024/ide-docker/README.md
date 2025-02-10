# Dockerized IDEs

TODO: standardized `./{ide}.sh {port}` scripts.

Stop running containers

```bash
docker stop $(docker ps -a -q)
```

## JupyterLab

```bash
./jupyterlab.sh 8880
```

https://stackoverflow.com/questions/47492150/how-do-i-set-a-custom-token-for-a-jupyter-notebook#comment136106744_51105004

https://jupyter.org/install#jupyterlab

## RStudio

```bash
./rstudio.sh 8880
```

https://cran.rstudio.com/

https://github.com/dkmiller/tidbits/blob/56fc6c7496b1e66e6c7482146c9673053a7224b7/2023/kubernetes/examples/e2e/images/ui/Dockerfile#L2

https://posit.co/download/rstudio-server/

https://posit.co/download/rstudio-desktop/

## VS Code

```bash
./vscode.sh 8880
```

https://coder.com/docs/code-server/guide

https://coder.com/docs/code-server/FAQ#how-does-the-config-file-work
