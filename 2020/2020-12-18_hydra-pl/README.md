# Hydra + PyTorch Lightning

Get a good working example nicely pulling together:

- Hydra for configuration
- PyTorch Lightning for ML
- (Run in AML?)
- (Horovod for distributed training?)

# Running locally

First time without the `data.download=...`.

```
python train.py data.root=/src/tmp/data data.download=false
```

# Running in AML

```
git clean -xdf; az ml job create --file job.yml --query metadata.interaction_endpoints.studio --out tsv
```

## Roadmap

- [ ] Export model to ONNX periodically (end of batch?).

## Links

- [Lightning in 2 steps](https://pytorch-lightning.readthedocs.io/en/stable/new-project.html)
- [Train models (create jobs)](https://azure.github.io/azureml-v2-preview/_build/html/quickstart/jobs.html)
- [Distributed training](https://azure.github.io/azureml-v2-preview/_build/html/quickstart/distributed-training.html)
