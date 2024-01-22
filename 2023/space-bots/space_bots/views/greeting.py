from dataclasses import dataclass

import streamlit as st
from injector import inject

from space_bots.client import SpacebotsClient
from space_bots.models import User
from space_bots.plural import Plural


@inject
@dataclass
class Greeting:
    client: SpacebotsClient
    user: User
    plural: Plural

    def write(self):
        credits = self.plural.render("credit", self.user.credits)
        fleets = self.plural.render("fleet", len(self.client.fleets()))
        st.write(
            f"Welcome, **{self.user.name:.8}**! You have **{credits}** and **{fleets}**."
        )
