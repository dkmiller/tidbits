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

## Links

- [Lightning in 2 steps](https://pytorch-lightning.readthedocs.io/en/stable/new-project.html)
