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
        # Change state type to string when state is a list
        if isinstance(state, list):
            state = str(state)
        return self.V[state]

    def add_state(self, state):
        # Change state type to string when state is a list
        if isinstance(state, list):
            state = str(state)

        # For empty eligibility or state not in eligibility
        if state not in self.eligibility:
            self.eligibility[state] = 0

        # For empty V or state not in V
        if state not in self.V:
            self.V[state] = random.uniform(0, 0.01)

    def set_TD_error(self, r, state, new_state, game_over):
        self.TD_error = (
                r
                + discount_factor_critic * self.get_value(new_state) * (not game_over)
                - self.get_value(state)
        )

    def set_eligibility(self, state, value=None):
        # Change state type to string when state is a list
        if isinstance(state, list):
            state = str(state)

        if value is None:
            self.eligibility[state] *= discount_factor_critic * \
                                       eligibility_decay_critic
        else:
            self.eligibility[state] = value

    def reset_eligibilities(self):
        self.eligibility.clear()

    def set_value_for_state(self, state):
        # Change state type to string when state is a list
        if isinstance(state, list):
            state = str(state)

        self.V[state] += lr_critic * self.get_TD_error() * \
                         self.eligibility[state]
