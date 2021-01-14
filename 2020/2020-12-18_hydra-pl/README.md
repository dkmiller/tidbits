# Hydra + PyTorch Lightning

Get a good working example nicely pulling together:

- Hydra for configuration
- PyTorch Lightning for ML
- (Run in AML?)
- (Horovod for distributed training?)

# Running locally

Run `Submit.ps1 -Local`.

# Running in AML

Run `Submit.ps1`, optionally with `-File sweep`.

## Roadmap

- [ ] Export model to ONNX periodically.
    - Ideally, do this at the end of each batch.
    - Currently encountering "maximum recursion depth exceeded" when calling the
      built-in `to_onnx` method.

## Links

- [Lightning in 2 steps](https://pytorch-lightning.readthedocs.io/en/stable/new-project.html)
- [Train models (create jobs)](https://azure.github.io/azureml-v2-preview/_build/html/quickstart/jobs.html)
- [Distributed training](https://azure.github.io/azureml-v2-preview/_build/html/quickstart/distributed-training.html)
- [azuremlv2 &gt; danmill](https://ml.azure.com/experiments/id/2eeb53e6-245e-4ec3-b52a-5be72ec5f20c?wsid=/subscriptions/48bbc269-ce89-4f6f-9a12-c6f91fcb772d/resourcegroups/aml1p-rg/workspaces/aml1p-ml-wus2&tid=72f988bf-86f1-41af-91ab-2d7cd011db47#21bc4a56-e69f-46fd-b141-8c08f8a616a8)
- [PyTorch Lightning &gt; Distributed modes](https://pytorch-lightning.readthedocs.io/en/latest/multi_gpu.html#distributed-modes)
- [Sweep jobs (hyperparameter tuning)](https://azure.github.io/azureml-v2-preview/_build/html/quickstart/jobs.html#sweep-jobs-hyperparameter-tuning)
- [Feature 788429: Support for torch.distributed.launch](https://dev.azure.com/msdata/Vienna/_workitems/edit/788429)
- [PyTorch Lightning ddp accelerator (Per-Node-Launch)](https://azure.github.io/azureml-web/docs/cheatsheet/distributed-training#pytorch-lightning-ddp-accelerator-per-node-launch)
- [Multi-node distributed training with PyTorch Lightning](https://github.com/Azure/azureml-examples/blob/main/tutorials/using-pytorch-lightning/4.train-multi-node-ddp.ipynb)
- [e2edemos &gt; tfv2-mnist-example &gt; Run 1](https://ml.azure.com/experiments/id/551a6619-86f5-4eec-9a66-ed354f13940f/runs/tfv2-mnist-example_1598911924_4a281f7b?wsid=/subscriptions/92c76a2f-0e1c-4216-b65e-abf7a3f34c1e/resourcegroups/demorg/workspaces/e2edemos&tid=72f988bf-86f1-41af-91ab-2d7cd011db47)
