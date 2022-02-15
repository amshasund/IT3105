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
        # print("Add state", state)
        # Change state type to string when state is a list
        if isinstance(state, list):
            state = str(state)

        # For empty eligibility
        if not self.eligibility:
            self.eligibility[state] = 0

        # For state not in eligibility
        elif state not in list(self.eligibility.keys()):
            self.eligibility[state] = 0

        # For empty V# if not state in self.V
        if not self.V:
            # TODO: litt høye verdier -> 0 og 0.01 (har litt med reward å gjøre)
            self.V[state] = random.uniform(0, 0.01)

        # For state not in V
        elif state not in list(self.V.keys()):
            # TODO: Not round
            self.V[state] = random.uniform(0, 0.01)

    # Ta med is game_over inn i TD error og if true sett value : 0
    def set_TD_error(self, r, state, new_state, game_over):
        self.TD_error = (
                r
                + discount_factor_critic * self.get_value(new_state) * (not game_over)
                - self.get_value(state)
        )
        # print(self.TD_error)

    def set_eligibility(self, state, value=None):
        # Change state type to string when state is a list
        if isinstance(state, list):
            state = str(state)
        if value is None:
            # TODO: Not round
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

        # TODO: Not round
        self.V[state] += lr_critic * self.get_TD_error() * \
                         self.eligibility[state]
