# Webscraping

Inspiration:
[Extract all the URLs from the webpage Using Python](https://www.geeksforgeeks.org/extract-all-the-urls-from-the-webpage-using-python/)

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

[Conflict-free replicated data type](https://en.wikipedia.org/wiki/Conflict-free_replicated_data_type)

## Help

[Unable to import psutil on M1 mac with miniforge: (`mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64e')`)](https://stackoverflow.com/a/72619209)

[ImportError: cannot import name 'generator_to_async_generator'](https://github.com/ipython/ipython/issues/11270#issuecomment-427448691)

:confused: [Remote jupyterhub kernel fails to launch: "waiting for kernel to be idle"](https://github.com/microsoft/vscode-jupyter/issues/10637)

[Unable to start kernel - Waiting for Jupiter Session to be Idle (Kernel fails to start as completions request crahes without sending a response back)](https://github.com/microsoft/vscode-jupyter/issues/9014)

[Case insensitive regular expression without `re.compile`?](https://stackoverflow.com/a/10444271)
