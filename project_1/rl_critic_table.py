import random
from parameters import (
    lr_critic,
    discount_factor_critic,
    eligibility_decay_critic
)


class CriticTable:
    def __init__(self, sim_world):
        self.sim_world = sim_world
        self.V = dict()
        self.eligibility = dict()
        self.TD_error = 0

    def get_state(self):
        return self.sim_world.get_state()

    def get_TD_error(self):
        return self.TD_error

    def get_value(self, state):
        return self.V[state]

    def add_state(self, state):
        if state not in self.eligibility.keys():
            self.eligibility[state] = 0
        if state not in self.V.keys():
            self.V[state] = round(random.uniform(0, 2), 3)

    def set_TD_error(self, r, state, new_state):
        self.TD_error = (
            r
            + discount_factor_critic * self.get_value(new_state)
            - self.get_value(state)
        )

    def set_eligibility(self, state, value=None):
        if value is None:
            self.eligibility[state] *= discount_factor_critic * \
                eligibility_decay_critic
        else:
            self.eligibility[state] = value

    def set_value_for_state(self, state):
        self.V[state] += lr_critic * self.get_TD_error() * \
            self.eligibility[state]
