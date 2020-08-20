# PyTorch Lightning

Learn
[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning),
following
[From PyTorch to PyTorch Lightning &mdash; A gentle introduction](https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09).

## Environment

```powershell
# Create environment.
conda env create --file environment.yml

# Activate environment.
conda activate pt-lightning

# Update environment.
conda env update --name pt-lightning --file environment.yml

# (Optional) cleanup environment.
conda remove --name pt-lightning --all --yes
```

## Links

- [MNIST hello world](https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=gEulmrbxwaYL)
- [Quick start](https://pytorch-lightning.readthedocs.io/en/stable/new-project.html)
- [`pytorch_mnist.py`](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/ml-frameworks/pytorch/training/distributed-pytorch-with-nccl-gloo/pytorch_mnist.py)
- [`loggers/comet.py`](https://github.com/PyTorchLightning/PyTorch-Lightning/blob/master/pytorch_lightning/loggers/comet.py)
- [Set up and use compute targets for model training](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
