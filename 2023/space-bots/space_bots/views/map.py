from dataclasses import dataclass

import plotly.graph_objects as go
import streamlit as st
from injector import inject

from space_bots.client import SpacebotsClient
from space_bots.models import Config, Fleet, System


def _render_system(system: System) -> str:
    data = system.name
    asteroid = system.asteroid
    if asteroid:
        data += " (🪨 " + " ".join(asteroid.values()) + " )"
    if system.station.get("directSell"):
        data += " 💰"
    return data


@st.cache_resource
def _plot(config: Config, fleets: list[Fleet], systems: dict[str, System]):
    """
    https://plotly.com/python/network-graphs/
    """
    edge_x = []
    edge_y = []
    node_x = []
    node_y = []
    customdata = []
    resources = []
    sizes = []

    fleet_locations = set(fleet.location_system_id for fleet in fleets)
    for system_id, system in systems.items():
        if system_id in fleet_locations:
            sizes.append(30)
        else:
            sizes.append(15)
        node_x.append(system.x)
        node_y.append(system.y)
        data = _render_system(system)
        asteroid = system.asteroid
        resources.append(len(asteroid.values()))
        for fleet in fleets:
            if system_id == fleet.location_system_id:
                data += f" 🚀 {fleet.id:.8}"

        customdata.append(data)
        for neighbor_ref in system.neighboring_systems:
            # Edge: system --> neighbor.
            neighbor = systems[neighbor_ref]
            edge_x.append(system.x)
            edge_x.append(neighbor.x)
            edge_x.append(None)
            edge_y.append(system.y)
            edge_y.append(neighbor.y)
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
        text=customdata,
        # https://plotly.com/python/hover-text-and-formatting/#customizing-hover-text-with-a-hovertemplate
        hovertemplate="%{text}",
        marker=dict(
            showscale=True,
            # https://plotly.com/python/builtin-colorscales/
            colorscale=config.colorscale,
            color=resources,
            # TODO: why does non-constant size kill the border?
            size=sizes,
            colorbar=dict(
                thickness=15,
                title="Resources",
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


@inject
@dataclass
class Map:
    client: SpacebotsClient
    config: Config
    systems: dict[str, System]

    def write(self):
        plot = _plot(self.config, self.client.fleets(), self.systems)

        st.plotly_chart(plot)
