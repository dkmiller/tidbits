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

# Cleanup
kubectl delete deployments,ingress,pods,services --all
```

https://stackoverflow.com/a/52176544/

http://localhost:8001/ui

## Future

- [ ] Build \& tag Docker images as `${name}:${dirhash}`.

## Links

[`kubectl` Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)

[FastAPI &gt; First Steps](https://fastapi.tiangolo.com/tutorial/first-steps/)

[Kubernetes &gt; Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)

https://iximiuz.com/en/posts/kubernetes-kind-load-docker-image/

https://kind.sigs.k8s.io/docs/user/ingress/

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
