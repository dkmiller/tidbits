https://towardsdatascience.com/large-graph-visualization-tools-and-approaches-2b8758a1cd59

https://cytoscape.org/

```powershell
docker run --volume ${PWD}:/src python:3.8-slim-buster python /src/evil.py
```

Or with conda/miniconda3.

```ps
docker build -t evil .

docker run evil
```