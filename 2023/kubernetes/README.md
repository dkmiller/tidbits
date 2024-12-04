# Agile Kubernetes development

Using [kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation).

## Setup

Start by running these commands once to install the necessary OS-level tooling and the `ptah` CLI:

```bash
brew bundle install && pip install -e .

# https://github.com/roboll/helmfile/issues/1182#issuecomment-790499993
helm plugin install https://github.com/databus23/helm-diff
```

## Build & run

Then, to build and ship things:

```bash
cd examples/e2e

# Build Docker images + Kubernetes manifests
ptah build

# Push Docker images to Kind cluster + apply Kubernetes manifests
ptah ship

# Copy the "admin" access key, port-forward the Kubernetes dashboard, and open it.
ptah dash

# Cleanup (DESTRUCTIVE!)
ptah nuke --no-whatif
```

## Miscellaneous commands

Visiting the admin. First command creates a token and copies it to the clipboard.

```bash
kubectl -n kubernetes-dashboard create token admin-user | pbcopy && open 'http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/'
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

- [ ] Liveness probe takes screenshot
    - https://github.com/puppeteer/puppeteer/issues/4039
    - https://github.com/puppeteer/puppeteer/issues/1947
    - https://github.com/isholgueras/chrome-headless/issues/1

## Links

[Brewfile: a Gemfile, but for Homebrew](https://thoughtbot.com/blog/brewfile-a-gemfile-but-for-homebrew)

[`kubectl` Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)

[How to access/expose kubernetes-dashboard service outside of a cluster?](https://stackoverflow.com/a/52176544/)

[FastAPI &gt; First Steps](https://fastapi.tiangolo.com/tutorial/first-steps/)

[Kubernetes &gt; Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)

[KiND &mdash; How I Wasted a Day Loading Local Docker Images](https://iximiuz.com/en/posts/kubernetes-kind-load-docker-image/)

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

[From inside of a Docker container, how do I connect to the localhost of the machine?](https://stackoverflow.com/a/63112795)

[`pyppeteer.errors.BrowserError`: Browser closed unexpectedly](https://github.com/davidteather/TikTok-Api/issues/178#issuecomment-657244793)

[Use await in python REPL directly](https://stackoverflow.com/a/68218635)

https://github.com/magsther/awesome-opentelemetry

[Tried to use SessionInfo before it was initialized](https://github.com/streamlit/streamlit/issues/879)

https://community.grafana.com/t/error-connecting-otel-collector-and-tempo/72869/4

- Tempo can't collect metrics, only traces.

https://github.com/open-telemetry/opentelemetry-helm-charts/issues/562

[How to Integrate Prometheus and Grafana on Kubernetes Using Helm](https://semaphoreci.com/blog/prometheus-grafana-kubernetes-helm)

https://opentelemetry.io/docs/languages/sdk-configuration/general/#otel_metrics_exporter

https://opentelemetry.io/docs/languages/sdk-configuration/otlp-exporter/

https://horovits.medium.com/prometheus-now-supports-opentelemetry-metrics-83f85878e46a

https://github.com/kubernetes/dashboard/issues/2970
