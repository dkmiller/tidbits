https://kind.sigs.k8s.io/docs/user/quick-start/#installation

- https://thoughtbot.com/blog/brewfile-a-gemfile-but-for-homebrew

```bash
kind create cluster --config kind.yaml

docker build -t api:0.0.1 containers/api/
kind load docker-image api:0.0.1

# https://stackoverflow.com/a/59493623
kubectl apply -R -f k8s

# Cleanup
kubectl delete deployments,ingress,pods,services --all
```

https://fastapi.tiangolo.com/tutorial/first-steps/

https://kubernetes.io/docs/concepts/workloads/controllers/deployment/

https://iximiuz.com/en/posts/kubernetes-kind-load-docker-image/

https://kind.sigs.k8s.io/docs/user/ingress/

----

Ingress + Kind: https://kind.sigs.k8s.io/docs/user/ingress/

https://kind.sigs.k8s.io/docs/user/loadbalancer/

https://kubernetes.io/docs/concepts/services-networking/ingress/

https://www.tutorialworks.com/kubernetes-pod-ip/

https://kubernetes.io/docs/reference/kubectl/cheatsheet/

https://kubernetes.io/docs/tutorials/

https://kubernetes.io/docs/concepts/workloads/controllers/deployment/

https://kubernetes.io/docs/concepts/workloads/pods/#working-with-pods
