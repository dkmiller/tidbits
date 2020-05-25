# Natural Language Processing with PyTorch

From the O'Reilly book
[with the same name](https://learning.oreilly.com/library/view/natural-language-processing/9781491978221/).
Its' companion repository is
[PyTorchNLPBook](https://github.com/joosthub/PyTorchNLPBook).

Here is a PDF &mdash;
[MLResources](https://github.com/dlsucomet/MLResources/blob/master/books/%5BNLP%5D%20Natural%20Language%20Processing%20with%20PyTorch%20(2019).pdf)
That copy may be bootleg, i.e. I have no idea if the repositories owner has any
right to host it.

## Environment

Here are some commands for using Anaconda to create, update, and cleanup
an environment for running this code.

```powershell
# Create environment.
conda env create --file environment.yml

# Activate environment.
conda activate nlp-pytorch

# Update environment.
# https://stackoverflow.com/a/43873901
conda env update --name nlp-pytorch --file environment.yml

# (Optional) cleanup environment.
conda remove --name nlp-pytorch --all --yes
```

## Links

- [`bidict`](https://bidict.readthedocs.io/en/master/)
- [PyYAML `yaml.load(input)` Deprecation](https://msg.pyyaml.org/load)
- [`typing` &mdash; Support for type hints](https://docs.python.org/3/library/typing.html)
- [Managing environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
- [How to Prepare Text Data for Machine Learning with scikit-learn](https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/)
- [`sklearn.feature_extraction.text.CountVectorizer`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
- [TypeError: batch must contain tensors, numbers, dicts or lists; found object](https://discuss.pytorch.org/t/typeerror-batch-must-contain-tensors-numbers-dicts-or-lists-found-object/14665/3)
- [How do I transform a "SciPy sparse matrix" to a "NumPy matrix"?](https://stackoverflow.com/a/26577144)
- [Exporting a model from PyTorch to ONNX and running it using ONNX runtime](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html)
- [[Running on windows 10] cuda runtime error (30) : unknown error at ..\aten\src\THC\THCGeneral.cpp:87](https://github.com/pytorch/pytorch/issues/17108)
- [typeguard](https://typeguard.readthedocs.io/en/latest/) (compare
  [typecheck-decorator](https://pypi.org/project/typecheck-decorator/))
- [`torch.utils.tensorboard`](https://pytorch.org/docs/stable/tensorboard.html)
- [Conda UnsatisfiableError: The following specifications were found to be incompatible with your CUDA driver](https://github.com/KevinMusgrave/pytorch-metric-learning/issues/55)
- [Managing environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
- [How to use Tensorboard with PyTorch?](https://discuss.pytorch.org/t/how-to-use-tensorboard-with-pytorch/61852)
- [Tensorboard TypeError: __init__() got an unexpected keyword argument 'serialized_options'](https://stackoverflow.com/a/57842296)
- [V1.0.1, `nn.BCEWithLogitsLoss` returns negative loss, Sigmoid layer not deployed](https://discuss.pytorch.org/t/v1-0-1-nn-bcewithlogitsloss-returns-negative-loss-sigmoid-layer-not-deployed/57409)
- [Get total amount of free GPU memory and available using pytorch](https://stackoverflow.com/a/58216793)
- [PyTorch &mdash; Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)
