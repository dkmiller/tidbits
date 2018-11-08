# Spark

First steps with Apache Spark, following:

- https://github.com/gettyimages/docker-spark
- _[Learning Apache Spark 2](http://a.co/d/5X8Bb7b)_, by Muhammad Asif Abbasi.

You need the current version of Docker for Windows (&ge; `18.06.1-ce`) installed
and running.

## Running

Try running:

```powershell
docker-compose up

# In a separate terminal window:
docker exec -it docker-spark_master_1 /bin/bash
```

You can visit the Spark UI at http://localhost:8080, and see the jobs http://localhost:4040/jobs/.

From the Spark window:

```bash
./bin/spark-shell
```

## Troubleshooting

If you encounter:

> ERROR: client version 1.21 is too old. Minimum supported API version is 1.24, please upgrade your client to a newer version

then use the local copy of `docker-compose.yml` following:
[client version 1.22 is too old. Minimum supported API version is 1.25](https://github.com/docker/compose/issues/5103)

If `Run.ps1` is hanging, follow [this article](https://docs.docker.com/engine/reference/commandline/system_prune/)
and run

```powershell
docker system prune --force
```
