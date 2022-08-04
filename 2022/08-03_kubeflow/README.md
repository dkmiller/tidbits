# Kubeflow

[Local Deployment &gt; Creating a cluster on kind](https://www.kubeflow.org/docs/components/pipelines/installation/localcluster-deployment/#2-creating-a-cluster-on-kind)

[Building components](https://www.kubeflow.org/docs/components/pipelines/sdk/component-development/)

[Pipeline Parameters](https://www.kubeflow.org/docs/components/pipelines/sdk/parameters/)

## Setup

Ensure Docker is up and running.

```bash
brew install kind
kind create cluster

export PIPELINE_VERSION=1.8.3

kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=$PIPELINE_VERSION"

# Wait a while...
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
```

Visit: http://localhost:8080/

## Pipeline submission

First, log in following

[Working with the Container registry &gt; Authenticating to the Container registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry#authenticating-to-the-container-registry)

```
docker login ghcr.io -u DUMMY 
```

Then, ...

```bash
conda activate kubeflow

python pipelines/sample.py
```

## Links

[Error unmarshalling content: invalid character '<' looking for beginning of value.](https://github.com/moby/moby/issues/40419)

[Error on import KFP at the first time [appengine]](https://github.com/kubeflow/pipelines/issues/2862)

>  Failed to pull image "ghcr.io/dkmiller/tidbits/gen_data": rpc error: code
>  = Unknown desc = failed to pull and unpack image
>  "ghcr.io/dkmiller/tidbits/gen_data:latest": failed to resolve reference
>  "ghcr.io/dkmiller/tidbits/gen_data:latest": failed to authorize: failed to
> fetch anonymous token: unexpected status: 401 Unauthorized

https://stackoverflow.com/questions/56494402/

[TFJob should work well with pipelines](https://github.com/kubeflow/pipelines/issues/677)

[Add pipeline launcher components for other distributed training jobs](https://github.com/kubeflow/pipelines/issues/3445)

https://linuxhint.com/kubectl-get-list-namespaces/

https://github.com/kubeflow/pytorch-operator/issues/330
