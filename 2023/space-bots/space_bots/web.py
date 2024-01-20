import plotly.graph_objects as go
import streamlit as st
import streamlit_pydantic as sp
import time
import inflect
import humanize

from space_bots.client import SpacebotsClient
from space_bots.config import Config

st.set_page_config("Space Bots", page_icon=":rocket:")


st.title("[Space Bots](https://space-bots.longwelwind.net/)")

p = inflect.engine()


def _pluralize(word, num):
    if num == 1:
        return f"{num} {word}"
    return f"{num} {p.plural(word)}"


with st.expander("Configuration"):
    config = sp.pydantic_form(key="config", model=Config) or Config()


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


def _render_system(system: dict) -> str:
    data = system["name"]
    asteroid = system.get("asteroid", {})
    if asteroid:
        data += " (🪨 " + " ".join(asteroid.values()) + " )"
    if system.get("station", {}).get("directSell"):
        data += " 💰"
    return data


@st.cache_data
def plot_traversal(config: Config, traversal: dict[str, dict], fleets: list[dict]):
    # https://plotly.com/python/network-graphs/
    edge_x = []
    edge_y = []
    node_x = []
    node_y = []
    customdata = []
    resources = []
    sizes = []

    fleet_locations = set(f["locationSystemId"] for f in fleets)

    for system_id, system in traversal.items():
        if system_id in fleet_locations:
            sizes.append(30)
        else:
            sizes.append(15)
        node_x.append(system["x"])
        node_y.append(system["y"])
        data = _render_system(system)
        asteroid = system.get("asteroid", {})
        resources.append(len(asteroid.values()))
        for fleet in fleets:
            if system_id == fleet["locationSystemId"]:
                data += f" 🚀 {fleet['id']:.8}"


        customdata.append(data)
        for neighbor_ref in system["neighboringSystems"]:
            # Edge: system --> neighbor.
            neighbor = traversal[neighbor_ref["systemId"]]
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


spacebots = SpacebotsClient(token=config.token)

user = spacebots.users()
fleets = spacebots.fleets()

_credits = _pluralize("credit", user['credits'])
_fleets = _pluralize("fleet", len(fleets))
st.write(f"Welcome, **{user['name']:.8}**! You have **{_credits}** and **{_fleets}**.")


all_systems = traverse(spacebots, fleets[0]["locationSystemId"])
plot = plot_traversal(config, all_systems, fleets)

st.plotly_chart(plot)



actions = ["Travel", "Transfer ships"]

# TODO: mine if possible

if any(sum(fleet.get("ships", {}).values()) < 100 for fleet in fleets):
    actions += ["Buy ships"]

mineable_fleets = [fleet for fleet in fleets if all_systems[fleet["locationSystemId"]].get("asteroid")]

if mineable_fleets:
    actions += ["Mine"]

sellable_fleets = [fleet for fleet in fleets if fleet.get("cargo") and all_systems[fleet["locationSystemId"]].get("station", {}).get("directSell")]

if sellable_fleets:
    actions += ["Sell"]


action = st.radio("Action", actions)


if action == "Travel":
    movement = {}
    for fleet in fleets:
        fleet_id = fleet["id"]
        system_id = fleet["locationSystemId"]
        neighbors = all_systems[system_id]["neighboringSystems"]
        status_quo = f"Sit still {_render_system(all_systems[system_id])}"
        destination = st.radio(
            f"{fleet_id:.8} &mapsto;",
            [status_quo] + [
                _render_system(all_systems[ns["systemId"]])
                for ns in neighbors
            ],
            key=f"destination-{fleet_id}",
            horizontal=True,
        )
        if destination == status_quo:
            continue
        destination_id = next(
            ns["systemId"]
            for ns in neighbors
            if all_systems[ns["systemId"]]["name"] in destination
        )
        movement[fleet_id] = destination_id

    if movement and st.button(f":arrow_right: Travel {movement}"):
        prog = st.progress(0.0, "Traveling ...")
        traveled = 0.0
        for fleet_id, destination_id in movement.items():
            spacebots.travel(fleet_id, destination_id)
            traveled += 1 / len(movement)
            prog.progress(traveled, f"Traveled {traveled:.0%} ...")
        prog.empty()
        st.experimental_rerun()

    #     _go = st.button("Travel", key=f"travel-{fleet_id}")
    #     if _go:
    #         destination_id = next(
    #             ns["systemId"]
    #             for ns in all_systems[system_id]["neighboringSystems"]
    #             if all_systems[ns["systemId"]]["name"] == destination
    #         )
    #         result = spacebots.travel(fleet_id, destination_id)
    #         st.write(result)
    #         st.experimental_rerun()

    # pass
elif action == "Buy ships":
    ship_types = spacebots.ship_types()
    ship_type = st.radio("Ship type", [s["id"] for s in ship_types])
    ship_price = [s["price"] for s in ship_types if s["id"] == ship_type][0]

    max_buy = min(100 - total_ships, int(user['credits'] / ship_price))
    n_ships = st.select_slider("Number of ships", range(1, max_buy + 1), max_buy)
    _go = st.button("🚀 Buy ships")
    if _go:
        with st.spinner(f"🚀 Buying {n_ships} ships..."):
            result = spacebots.buy_ships(fleet_id, {ship_type: n_ships})
            st.write(result)
            st.experimental_rerun()
elif action == "Mine":
    _fl = " + ".join(map(lambda x: f"{x['id']:.8} {all_systems[x['locationSystemId']].get('asteroid')['miningResourceId']}", mineable_fleets))
    if st.button(f"🪨 Mine ({_fl})"):
        index = 1
        prog = st.progress(0, f"🪨 Mining for {humanize.ordinal(index)} time ...")
        while index:
            sleep = 0
            for fleet in mineable_fleets:
                result = spacebots.mine(fleet["id"])
                sleep = max(sleep, result.get("duration", 0))
            index += 1
            prog.progress(index, f"🪨 Mining for {humanize.ordinal(index)} time ...")
            if sleep:
                with st.spinner(f"⏳ Waiting for {sleep} seconds..."):
                    time.sleep(sleep)
        st.experimental_rerun()
elif action == "Sell" and st.button("💰 Sell"):
    with st.spinner("💰 Selling..."):
        for fleet in sellable_fleets:
            spacebots.sell(fleet["id"], fleet["cargo"])
        st.experimental_rerun()

st.stop()

for fleet in fleets:
    fleet_id = fleet["id"]
    msg = f"Fleet {fleet_id:.8}"
    ships = fleet.get("ships")
    cargo = fleet.get("cargo", {})
    if cargo or ships:
        msg += " ("
    if ships:
        msg += ", ".join(map(lambda x: _pluralize(x[0], x[1]), ships.items()))
    if ships and cargo:
        msg += ", "
    if cargo:
        msg += " + ".join(map(lambda x: _pluralize(x[0], x[1]), cargo.items()))
    if cargo or ships:
        msg += ")"
    st.write(msg)

    # st.write(fleet)
    system_id = fleet["locationSystemId"]
    all_systems = traverse(spacebots, system_id)
    plot = plot_traversal(config, all_systems, all_fleets)
    st.plotly_chart(plot)

    # for resource in cargo.keys():
    #     st.write(resource)
    #     for system_id in all_systems.keys():
    #         market = spacebots.market(system_id, resource)
    #         st.write(market)

    actions = ["Travel", "Mine", "Transfer ships"]

    if ships:
        total_ships = sum(ships.values())
        if total_ships < 100:
            actions += ["Buy ships"]


    if all_systems[system_id].get("station", {}).get("directSell"):
        actions += ["Sell"]

    action = st.radio("Action", actions, key=f"action-{fleet_id}")
    if action == "Travel":
        destination = st.radio(
            "Destination",
            [
                all_systems[ns["systemId"]]["name"]
                for ns in all_systems[system_id]["neighboringSystems"]
            ],
            key=f"destination-{fleet_id}",
        )
        _go = st.button("Travel", key=f"travel-{fleet_id}")
        if _go:
            destination_id = next(
                ns["systemId"]
                for ns in all_systems[system_id]["neighboringSystems"]
                if all_systems[ns["systemId"]]["name"] == destination
            )
            result = spacebots.travel(fleet_id, destination_id)
            st.write(result)
            st.experimental_rerun()
    elif action == "Mine":
        _go = st.button("Mine")
        if _go:
            index = 1
            prog = st.progress(0, f"🪨 Mining for {humanize.ordinal(index)} time ...")
            while index:
                result = spacebots.mine(fleet_id)
                index += 1
                prog.progress(index, f"🪨 Mining for {humanize.ordinal(index)} time ...")
                sleep = result.get("duration", 0)
                if sleep:
                    with st.spinner(f"⏳ Waiting for {sleep} seconds..."):
                        time.sleep(sleep)
            st.experimental_rerun()
    elif action == "Buy ships":
        ship_types = spacebots.ship_types()
        ship_type = st.radio("Ship type", [s["id"] for s in ship_types])
        ship_price = [s["price"] for s in ship_types if s["id"] == ship_type][0]

        max_buy = min(100 - total_ships, int(user['credits'] / ship_price))
        n_ships = st.select_slider("Number of ships", range(1, max_buy + 1), max_buy)
        _go = st.button("🚀 Buy ships")
        if _go:
            with st.spinner(f"🚀 Buying {n_ships} ships..."):
                result = spacebots.buy_ships(fleet_id, {ship_type: n_ships})
                st.write(result)
                st.experimental_rerun()
    elif action == "Transfer ships":
        ship_types = spacebots.ship_types()
        ship_type = st.radio("Ship type", [s["id"] for s in ship_types])
        n_type = ships[ship_type]
        n_ships = st.select_slider("Number of ships", range(1, n_type), n_type - 1)
        _go = st.button("Transfer")
        if _go:
            with st.spinner("?? Transfering..."):
                result = spacebots.transfer(fleet_id, {ship_type: n_ships})
                st.write(result)
    else:
        _go = st.button("Sell")
        if _go:
            with st.spinner("💰 Selling..."):
                result = spacebots.sell(fleet_id, cargo)
            st.write(result)
            st.experimental_rerun()

    st.write(fleet)
    st.write("---")
