import streamlit as st

from space_bots.builders import get
from space_bots.views import Game


# Push caching to the current page.
# @st.cache_data(ttl=60 * 5)
def _game():
    return get(Game)


st.set_page_config("Space Bots", page_icon=":rocket:")

game = _game()
game.write()

st.stop()


# actions = ["Travel", "Transfer ships"]

# # TODO: mine if possible

# if any(sum(fleet.get("ships", {}).values()) < 100 for fleet in fleets):
#     actions += ["Buy ships"]

# mineable_fleets = [fleet for fleet in fleets if all_systems[fleet["locationSystemId"]].get("asteroid")]

# if mineable_fleets:
#     actions += ["Mine"]

# sellable_fleets = [fleet for fleet in fleets if fleet.get("cargo") and all_systems[fleet["locationSystemId"]].get("station", {}).get("directSell")]

# if sellable_fleets:
#     actions += ["Sell"]


# action = st.radio("Action", actions)


# if action == "Travel":
#     movement = {}
#     for fleet in fleets:
#         fleet_id = fleet["id"]
#         system_id = fleet["locationSystemId"]
#         neighbors = all_systems[system_id]["neighboringSystems"]
#         status_quo = f"Sit still {_render_system(all_systems[system_id])}"
#         destination = st.radio(
#             f"{fleet_id:.8} &mapsto;",
#             [status_quo] + [
#                 _render_system(all_systems[ns["systemId"]])
#                 for ns in neighbors
#             ],
#             key=f"destination-{fleet_id}",
#             horizontal=True,
#         )
#         if destination == status_quo:
#             continue
#         destination_id = next(
#             ns["systemId"]
#             for ns in neighbors
#             if all_systems[ns["systemId"]]["name"] in destination
#         )
#         movement[fleet_id] = destination_id

#     if movement and st.button(f":arrow_right: Travel {movement}"):
#         prog = st.progress(0.0, "Traveling ...")
#         traveled = 0.0
#         for fleet_id, destination_id in movement.items():
#             spacebots.travel(fleet_id, destination_id)
#             traveled += 1 / len(movement)
#             prog.progress(traveled, f"Traveled {traveled:.0%} ...")
#         prog.empty()
#         st.experimental_rerun()

#     #     _go = st.button("Travel", key=f"travel-{fleet_id}")
#     #     if _go:
#     #         destination_id = next(
#     #             ns["systemId"]
#     #             for ns in all_systems[system_id]["neighboringSystems"]
#     #             if all_systems[ns["systemId"]]["name"] == destination
#     #         )
#     #         result = spacebots.travel(fleet_id, destination_id)
#     #         st.write(result)
#     #         st.experimental_rerun()

#     # pass
# elif action == "Buy ships":
#     ship_types = spacebots.ship_types()
#     ship_type = st.radio("Ship type", [s["id"] for s in ship_types])
#     ship_price = [s["price"] for s in ship_types if s["id"] == ship_type][0]

#     max_buy = min(100 - total_ships, int(user['credits'] / ship_price))
#     n_ships = st.select_slider("Number of ships", range(1, max_buy + 1), max_buy)
#     _go = st.button("🚀 Buy ships")
#     if _go:
#         with st.spinner(f"🚀 Buying {n_ships} ships..."):
#             result = spacebots.buy_ships(fleet_id, {ship_type: n_ships})
#             st.write(result)
#             st.experimental_rerun()
# elif action == "Mine":
#     _fl = " + ".join(map(lambda x: f"{x['id']:.8} {all_systems[x['locationSystemId']].get('asteroid')['miningResourceId']}", mineable_fleets))
#     if st.button(f"🪨 Mine ({_fl})"):
#         index = 1
#         prog = st.progress(0, f"🪨 Mining for {humanize.ordinal(index)} time ...")
#         while index:
#             sleep = 0
#             for fleet in mineable_fleets:
#                 result = spacebots.mine(fleet["id"])
#                 sleep = max(sleep, result.get("duration", 0))
#             index += 1
#             prog.progress(index, f"🪨 Mining for {humanize.ordinal(index)} time ...")
#             if sleep:
#                 with st.spinner(f"⏳ Waiting for {sleep} seconds..."):
#                     time.sleep(sleep)
#         st.experimental_rerun()
# elif action == "Sell" and st.button("💰 Sell"):
#     with st.spinner("💰 Selling..."):
#         for fleet in sellable_fleets:
#             spacebots.sell(fleet["id"], fleet["cargo"])
#         st.experimental_rerun()

# st.stop()

# for fleet in fleets:
#     fleet_id = fleet["id"]
#     msg = f"Fleet {fleet_id:.8}"
#     ships = fleet.get("ships")
#     cargo = fleet.get("cargo", {})
#     if cargo or ships:
#         msg += " ("
#     if ships:
#         msg += ", ".join(map(lambda x: _pluralize(x[0], x[1]), ships.items()))
#     if ships and cargo:
#         msg += ", "
#     if cargo:
#         msg += " + ".join(map(lambda x: _pluralize(x[0], x[1]), cargo.items()))
#     if cargo or ships:
#         msg += ")"
#     st.write(msg)

#     # st.write(fleet)
#     system_id = fleet["locationSystemId"]
#     all_systems = traverse(spacebots, system_id)
#     plot = plot_traversal(config, all_systems, all_fleets)
#     st.plotly_chart(plot)

#     # for resource in cargo.keys():
#     #     st.write(resource)
#     #     for system_id in all_systems.keys():
#     #         market = spacebots.market(system_id, resource)
#     #         st.write(market)

#     actions = ["Travel", "Mine", "Transfer ships"]

#     if ships:
#         total_ships = sum(ships.values())
#         if total_ships < 100:
#             actions += ["Buy ships"]


#     if all_systems[system_id].get("station", {}).get("directSell"):
#         actions += ["Sell"]

#     action = st.radio("Action", actions, key=f"action-{fleet_id}")
#     if action == "Travel":
#         destination = st.radio(
#             "Destination",
#             [
#                 all_systems[ns["systemId"]]["name"]
#                 for ns in all_systems[system_id]["neighboringSystems"]
#             ],
#             key=f"destination-{fleet_id}",
#         )
#         _go = st.button("Travel", key=f"travel-{fleet_id}")
#         if _go:
#             destination_id = next(
#                 ns["systemId"]
#                 for ns in all_systems[system_id]["neighboringSystems"]
#                 if all_systems[ns["systemId"]]["name"] == destination
#             )
#             result = spacebots.travel(fleet_id, destination_id)
#             st.write(result)
#             st.experimental_rerun()
#     elif action == "Mine":
#         _go = st.button("Mine")
#         if _go:
#             index = 1
#             prog = st.progress(0, f"🪨 Mining for {humanize.ordinal(index)} time ...")
#             while index:
#                 result = spacebots.mine(fleet_id)
#                 index += 1
#                 prog.progress(index, f"🪨 Mining for {humanize.ordinal(index)} time ...")
#                 sleep = result.get("duration", 0)
#                 if sleep:
#                     with st.spinner(f"⏳ Waiting for {sleep} seconds..."):
#                         time.sleep(sleep)
#             st.experimental_rerun()
#     elif action == "Buy ships":
#         ship_types = spacebots.ship_types()
#         ship_type = st.radio("Ship type", [s["id"] for s in ship_types])
#         ship_price = [s["price"] for s in ship_types if s["id"] == ship_type][0]

#         max_buy = min(100 - total_ships, int(user['credits'] / ship_price))
#         n_ships = st.select_slider("Number of ships", range(1, max_buy + 1), max_buy)
#         _go = st.button("🚀 Buy ships")
#         if _go:
#             with st.spinner(f"🚀 Buying {n_ships} ships..."):
#                 result = spacebots.buy_ships(fleet_id, {ship_type: n_ships})
#                 st.write(result)
#                 st.experimental_rerun()
#     elif action == "Transfer ships":
#         ship_types = spacebots.ship_types()
#         ship_type = st.radio("Ship type", [s["id"] for s in ship_types])
#         n_type = ships[ship_type]
#         n_ships = st.select_slider("Number of ships", range(1, n_type), n_type - 1)
#         _go = st.button("Transfer")
#         if _go:
#             with st.spinner("?? Transfering..."):
#                 result = spacebots.transfer(fleet_id, {ship_type: n_ships})
#                 st.write(result)
#     else:
#         _go = st.button("Sell")
#         if _go:
#             with st.spinner("💰 Selling..."):
#                 result = spacebots.sell(fleet_id, cargo)
#             st.write(result)
#             st.experimental_rerun()

#     st.write(fleet)
#     st.write("---")
