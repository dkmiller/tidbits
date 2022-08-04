# Apache Airflow

[Docker image for Apache Airflow](https://airflow.apache.org/docs/docker-stack/index.html)

```powershell
docker run -v ${PWD}:/opt/airflow/dags -p 8080:8080 -it apache/airflow standalone
```

then visit http://localhost:8080 and log in

Also helpful:

- [Running Airflow locally](https://airflow.apache.org/docs/apache-airflow/stable/start/local.html).
- [Tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial.html)
