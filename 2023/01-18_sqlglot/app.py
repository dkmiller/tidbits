from sqlglot import parse_one
from sqlglot.expressions import Expression
import streamlit as st
from streamlit_ace import st_ace


def name(e: Expression):
    prefix = "z_"
    return f"{prefix}_{id(e)}_{type(e).__name__}"


def visit(expr: Expression):
    __lines = []

    def _prune(item, parent, key):
        # print(f"I'm at {name(item)}")
        print(f'{name(item)}[label="({type(item).__name__}) {item}"];')
        __lines.append(f'{name(item)}[label="({type(item).__name__}) {item}"];')
        if parent:
            __lines.append(f"{name(parent)} -- {name(item)};")
            print(f"{name(parent)} -- {name(item)};")
        #     print(f"\tparent = {name(parent)}")
        # if key:
        #     print(f"\tkey = {key}")

    list(expr.bfs(prune=_prune))
    body = "\n".join(__lines)
    return f"""
graph {{
  {body}
}}
    """
    # return False
    # print(f"{expr.alias_or_name or id(expr)} {type(expr).__name__}")
    # for x in tree.bfs(prune=_prune):
    #     print(f"---> {type(x[0])}")
    # visit(x[0])
    #     print(x)
    # print(len(expr.expressions))
    # for e in expr.expressions:
    #     visit(e)


# tree = parse_one("SELECT a + 1 AS z")
# # print(tree.alias_or_name)
# # print(type(tree).__name__)
# # print(list(map(type,tree.expressions)))

# visit(tree)


# print(repr(tree))

# Spawn a new Ace editor
content = st_ace(value="SELECT a + 1 AS z", language="sql")

tree = parse_one(content)

x = visit(tree)

st.graphviz_chart(x)
