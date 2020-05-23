# Natural Language Processing with PyTorch

From the O'Reilly book
[with the same name](https://learning.oreilly.com/library/view/natural-language-processing/9781491978221/).

## Environment

Here are some commands for using Anaconda to create, update, and cleanup
an environment for running this code.

```powershell
# Create environment.
conda env create --file environment.yml

# Activate environment.
conda activate nlp

# Update environment.
# https://stackoverflow.com/a/43873901
conda env update --name nlp --file environment.yml

# (Optional) cleanup environment.
conda remove --name nlp --all --yes
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
