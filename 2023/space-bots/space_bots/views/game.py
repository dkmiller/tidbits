from dataclasses import dataclass

import streamlit as st
from injector import inject

from space_bots.models import Action
from space_bots.views.greeting import Greeting
from space_bots.views.map import Map


@inject
@dataclass
class Game:
    actions: list[Action]
    greeting: Greeting
    map: Map

    def write(self):
        st.title("[Space Bots](https://space-bots.longwelwind.net/)")
        self.greeting.write()

        if action := st.radio("Action", self.actions, format_func=lambda a: a.name, horizontal=True):
            if option := action.submenu():
                action.act(option)
            # if action.submenu():
            #     pass
                        #    [a.name for a in self.actions], horizontal=True)


        self.map.write()
