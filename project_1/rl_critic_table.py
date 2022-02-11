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

    def initialize_value_function(self, all_states):
        for s in all_states:
            self.V[s] = random.randint(0, 10)

    # should this be done differently?
    def initialize_eligibility_function(self, all_states):
        for s in all_states:
            self.eligibility[s] = 0

    def set_TD_error(self, r, state, new_state):
        self.TD_error = (
            r
            + discount_factor_critic * self.get_value(new_state)
            - self.get_value(state)
        )

    def set_eligibility(self, state, value):
        if value is None:
            self.eligibility[state] *= discount_factor_critic * \
                eligibility_decay_critic
        else:
            self.eligibility[state] = value

    def set_value_for_state(self, state):
        self.V[state] += lr_critic * self.get_TD_error() * \
            self.eligibility[state]
