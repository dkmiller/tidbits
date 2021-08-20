# Graph learning

Deck:
[Graph neural networks](https://microsoft-my.sharepoint.com/:p:/p/amsaied/EQZksGBlY65BhbZGhzZw7jcBXOCDGVqcIioEkA3zNcgQNg?e=50sqRB).

## Notes

Graph (undirected or DAG) &mapsto; one of many embeddings

Some feature(s) / node

GNN ... transform (feature / node) into (new feature / node)

Can do this many time ... get a complicated neural network (at each layer, everything is indexed on the graph).

Graph with only one node and a loop == "standard" (vanilla) neural network.

Typical use case: structure of graph is fixed, the thing you're learning is the weight matrices.

No "fundamentally new" ability, more conceptual simplification and allow smaller graphs.

E.g., 10k people == 10k features. But need a **much larger** "naive" graph to embed the graph.

Paper: _graph attention networks_.

:warning: Installation is difficult: https://github.com/rusty1s/pytorch_geometric/issues/1080%E2%80%8B

- Amin's recommendation: install from **source**.


