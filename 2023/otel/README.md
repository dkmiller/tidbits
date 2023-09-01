# FastAPI + OpenTelemetry

Follow links:

- https://www.cncf.io/blog/2022/04/22/opentelemetry-and-python-a-complete-instrumentation-guide/
- https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/fastapi/fastapi.html
- https://opentelemetry.io/docs/instrumentation/python/getting-started/
- https://opentelemetry.io/docs/instrumentation/python/automatic/

Run `./run.sh`, then visit:

> http://localhost:8000/roll?sides=2&rolls=6
> http://localhost:8000/fail?value=10

Looks like FastAPI doesn't have good auto instrumentation for exceptions yet:
https://medium.com/humanmanaged/quick-start-to-integrating-opentelemetry-with-fastapi-part-1-2be4fec874bc
