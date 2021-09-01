import mlflow


print("Hello, world!")

mlflow.autolog()

mlflow.log_metric("metric", 42.0)

print("Bye, world...")


