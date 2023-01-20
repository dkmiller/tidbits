from typing import Tuple

import streamlit as st
from sqlglot import parse_one
from sqlglot.expressions import Expression
from streamlit_ace import st_ace
from streamlit_agraph import Config, Edge, Node, agraph


def name(e: Expression):
    prefix = "z_"
    return f"{prefix}_{id(e)}_{type(e).__name__}"


def visit(expr: Expression):
    lines = set()

    def prune(item, parent, _):
        lines.add(f'{name(item)}[label="({type(item).__name__}) {item}"];')
        if parent:
            lines.add(f"{name(parent)} -- {name(item)};")

    list(expr.bfs(prune=prune))
    body = "\n".join(lines)
    return f"""
graph {{
  {body}
}}
    """


# Spawn a new Ace editor
content = st_ace(value="SELECT a + 1 AS z", language="sql")

tree = parse_one(content)

x = visit(tree)

st.graphviz_chart(x)


def to_agraph(e: Expression) -> Tuple[set, set]:
    nodes = set()
    edges = set()

    def prune(item, parent, _):
        nodes.add(Node(id=name(item), label=f"({type(item).__name__}) {item}"))
        if parent:
            edges.add(Edge(source=name(parent), target=name(item)))

    list(e.bfs(prune=prune))
    return nodes, edges


nodes, edges = to_agraph(tree)

agraph(nodes=nodes, edges=edges, config=Config())
