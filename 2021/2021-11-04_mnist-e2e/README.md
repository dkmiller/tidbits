# MNIST end-to-end in Azure ML

Specifically, use the CLI v2.

## Feedback

The statement "The extension will automatically install the first time you run
an `az ml` command."
[here](https://docs.microsoft.com/en-us/cli/azure/ml?view=azure-cli-latest)
is false.

`az ml *` command pages
([example](https://docs.microsoft.com/en-us/cli/azure/ml/data?view=azure-cli-latest#az_ml_data_create))
don't link to corresponding YAML schema specs.

`az ml component list` hangs indefinitely

~~No schema~~ file for component YMLs.
- Very difficult to find it

Intellisense for pipeline YAMLs sucks.

Snapshot upload is horribly slow.

WTF is this:

> Message: input is not a valid input name per component definition

Named datasets are lost in the pipelines UI.

CLI submission is **very** slow compared to component SDK.

No reuse :/

## Links

- https://marketplace.visualstudio.com/items?itemName=ms-toolsai.vscode-ai
- http://yann.lecun.com/exdb/mnist/
- https://docs.microsoft.com/en-us/azure/machine-learning/reference-yaml-dataset
- https://pypi.org/project/idx2numpy/
- https://docs.microsoft.com/en-us/azure/machine-learning/concept-component
- https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-component-pipelines-cli
- https://docs.microsoft.com/en-us/azure/machine-learning/reference-yaml-job-pipeline
- https://github.com/Azure/azureml-examples/blob/main/cli/jobs/pipelines-with-components/nyc_taxi_data_regression/job.yml
- https://www.pytorchlightning.ai/
