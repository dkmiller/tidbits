# Azure ML Workshop

Follow https://github.com/Azure-Samples/AzureMLWorkshop .

You should have
[PowerShell 7.0](https://devblogs.microsoft.com/powershell/announcing-powershell-7-0/)
installed.

Run locally

```ps
.\Bootstrap.ps1

python ./2.a/train.py --input_dir ./dsdevops-oh-files --output_dir .
```

Run globally

```ps
python ./2.a/experiment.py
```

## Links

- [Create Azure Machine Learning datasets](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-register-datasets)
- [How to use EstimatorStep in AML Pipeline](https://aka.ms/pl-estimator)
- [Monitor Azure ML experiment runs and metrics](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-track-experiments)
- [Convert a pipeline with a LightGbm model](http://onnx.ai/sklearn-onnx/auto_examples/plot_pipeline_lightgbm.html)
- [How to use `glob()` to find files recursively?](https://stackoverflow.com/a/2186565)
