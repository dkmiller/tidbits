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
    - [run-tutorial-upl \#996 :heavy_check_mark:](https://github.com/Azure/azureml-examples/runs/1709068525?check_suite_focus=true)
    - [pt-lightning-ddp-tutorial &gt; Run 719 :heavy_check_mark:](https://ml.azure.com/experiments/id/c719a3fe-ab85-4796-8f5d-2a089ed2f107/runs/pt-lightning-ddp-tutorial_1610722264_acb99097?wsid=/subscriptions/6560575d-fa06-4e7d-95fb-f962e74efd7a/resourcegroups/azureml-examples/workspaces/default&tid=72f988bf-86f1-41af-91ab-2d7cd011db47#outputsAndLogs)
- [e2edemos &gt; tfv2-mnist-example &gt; Run 1](https://ml.azure.com/experiments/id/551a6619-86f5-4eec-9a66-ed354f13940f/runs/tfv2-mnist-example_1598911924_4a281f7b?wsid=/subscriptions/92c76a2f-0e1c-4216-b65e-abf7a3f34c1e/resourcegroups/demorg/workspaces/e2edemos&tid=72f988bf-86f1-41af-91ab-2d7cd011db47)
- [Multi Node Distributed Training with PyTorch Lightning \& Azure ML](https://medium.com/microsoftazure/multi-node-distributed-training-with-pytorch-lightning-azure-ml-88ac59d43114)
- [(Torus) tnlrv3_continual_pretraining &gt; Run 63](https://ml.azure.com/experiments/id/bc0d0c25-cc31-4290-bd9d-d3fd8b9e9393/runs/73096fd6-abcd-43e3-8132-a527ca4f4af9?wsid=/subscriptions/98d476c6-c75b-481c-8c18-f53d91f614b0/resourcegroups/Phase2Prod/workspaces/Matrix&tid=cdc5aeea-15c5-4db6-b079-fcadd2505dc2#snapshot)
    - See: `biunilm/aml_env.py`.
