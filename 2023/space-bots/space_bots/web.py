import plotly.graph_objects as go
import streamlit as st

from space_bots.client import SpacebotsClient

st.set_page_config("Space Bots", page_icon=":rocket:")


st.title("[Space Bots](https://space-bots.longwelwind.net/)")


@st.cache_data(ttl=60 * 60)
def traverse(client: SpacebotsClient, start: str):
    visited = set()
    queue = [start]
    rv = {}
    while queue:
        current = queue.pop()
        if current in visited:
            continue
        visited.add(current)
        system = client.system(current)
        rv[current] = system
        queue.extend([x["systemId"] for x in system["neighboringSystems"]])
    return rv


def plot_traversal(traversal: dict[str, dict]):
    # https://plotly.com/python/network-graphs/
    edge_x = []
    edge_y = []
    node_x = []
    node_y = []

    for system_id, system in traversal.items():
        node_x.append(system["x"])
        node_y.append(system["y"])
        for neighbor_ref in system["neighboringSystems"]:
            # Edge: system --> neighbor.
            neighbor = traversal[neighbor_ref["systemId"]]
            # st.write(system)
            # st.write(neighbor)
            # st.write("---")
            edge_x.append(system["x"])
            edge_x.append(neighbor["x"])
            edge_x.append(None)
            edge_y.append(system["y"])
            edge_y.append(neighbor["y"])
            edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale="YlGnBu",
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right",
            ),
            line_width=2,
        ),
    )

    return go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title="Systems",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    # st.write(edge_x)


spacebots = SpacebotsClient()

for fleet in spacebots.fleets(user="my"):
    system_id = fleet["locationSystemId"]
    all_fleets = traverse(spacebots, system_id)
    plot = plot_traversal(all_fleets)
    # st.write(type(plot))
    st.plotly_chart(plot)

    st.write(fleet)
