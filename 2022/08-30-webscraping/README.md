# Webscraping

https://www.geeksforgeeks.org/extract-all-the-urls-from-the-webpage-using-python/

## Algo

Asynchronous, parallelized, fault-tolerant breadth-first search.

State:

- set of "visited nodes", only access under lock
- queue of "nodes to visit", only access under lock, initialize with starting node
- counter of "active visits", only access under lock, init to `0`

Methods:

- scrape
    - if empty queue + no active visits, done!
    - if empty queue + active visits, sleep ((configurable amount of time))
    - if non-empty queue, spawn parallel visit for everything in the queue
- visit(node)
  - if node already visited, return, otherwise add node to set
  - get target nodes (long-running)

## Alternatives

https://en.wikipedia.org/wiki/Conflict-free_replicated_data_type

## Help

https://stackoverflow.com/a/72619209

https://github.com/ipython/ipython/issues/11270#issuecomment-427448691

:/ https://github.com/microsoft/vscode-jupyter/issues/10637

https://github.com/microsoft/vscode-jupyter/issues/9014

[Case insensitive regular expression without `re.compile`?](https://stackoverflow.com/a/10444271)
