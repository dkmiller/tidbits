https://kind.sigs.k8s.io/docs/user/quick-start/#installation

- https://thoughtbot.com/blog/brewfile-a-gemfile-but-for-homebrew

```bash
kind create cluster --config kind.yaml

docker build -t api:0.0.1 containers/api/
docker build -t ui:0.0.4 containers/ui/

# "Ship" Docker images to local cluster.
kind load docker-image api:0.0.1 ui:0.0.4

# https://stackoverflow.com/a/59493623
kubectl apply -R -f k8s

# Port forward:
kubectl port-forward deployment/ui-deployment 8501:8501
kubectl port-forward deployment/api-deployment 8000:8000

# Cleanup
kubectl delete deployments,ingress,pods,services --all

docker system prune --all --force
```

https://stackoverflow.com/a/52176544/

Visiting the admin. First command creates a token and copies it to the clipboard.

```bash
kubectl -n kubernetes-dashboard create token admin-user | pbcopy
kubectl proxy
```

Pages:

- Dashboard: http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/
- UI: http://localhost:8501/
- API: http://localhost:8000/docs

Calling "localhost" from _inside_ a Docker image:

```bash
curl host.docker.internal:8000
```

## Future

- [ ] OpenTelemetry: https://www.reddit.com/r/kubernetes/comments/13ce38b/opentelemetry_visualization/
      https://github.com/magsther/awesome-opentelemetry
- [ ] Liveness probe takes screenshot
    - https://github.com/puppeteer/puppeteer/issues/4039
    - https://github.com/puppeteer/puppeteer/issues/1947
    - https://github.com/isholgueras/chrome-headless/issues/1

## Links

[`kubectl` Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)

[FastAPI &gt; First Steps](https://fastapi.tiangolo.com/tutorial/first-steps/)

[Kubernetes &gt; Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)

[KiND &mdash; How I Wasted a Day Loading Local Docker Images](https://iximiuz.com/en/posts/kubernetes-kind-load-docker-image/)

----

Ingress + Kind: https://kind.sigs.k8s.io/docs/user/ingress/

https://kind.sigs.k8s.io/docs/user/loadbalancer/

https://kubernetes.io/docs/concepts/services-networking/ingress/

https://www.tutorialworks.com/kubernetes-pod-ip/

https://kubernetes.io/docs/tutorials/

https://kubernetes.io/docs/concepts/workloads/controllers/deployment/

https://kubernetes.io/docs/concepts/workloads/pods/#working-with-pods

https://kubernetes.io/docs/tasks/access-application-cluster/web-ui-dashboard/

- https://stackoverflow.com/a/52176544

[Four-letter gods](https://anch.info/eng/fortuities/names/855/)

https://stackoverflow.com/a/63112795

https://github.com/davidteather/TikTok-Api/issues/178#issuecomment-657244793

[Use await in python REPL directly](https://stackoverflow.com/a/68218635)
