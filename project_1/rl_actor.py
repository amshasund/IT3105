import random
import copy
from parameters import (
    lr_actor,
    discount_factor_actor,
    eligibility_decay_actor
)


class Actor:
    def __init__(self, critic, sim_world):
        self.critic = critic
        self.sim_world = sim_world
        self.policy = dict()
        self.eligibility = dict()

    def get_best_action(self, state):
        if state not in self.policy.keys():
            return random.choice(
                self.sim_world.get_possible_actions_from_state(
                    state)
            )
        actions = self.policy[state]
        # TODO: shorten this?
        highest_value = max(actions.values())
        best_actions = []
        for key, value in actions.items():
            if value == highest_value:
                best_actions.append(key)
        return random.choice(best_actions)

    def add_state(self, state):
        if state not in self.eligibility.keys():
            self.eligibility[state] = dict()
            all_actions_for_state = \
                self.sim_world.get_possible_actions_from_state(state)
            for a in all_actions_for_state:
                self.eligibility[state][a] = 0
        if state not in self.policy.keys():
            self.policy[state] = dict.fromkeys(self.eligibility[state], 0)

    '''
    def initialize_policy_function(self, all_states):
        for s in all_states:
            self.policy[s] = dict()
            all_actions_for_state = self.sim_world.get_possible_actions_from_state(
                s)
            for a in all_actions_for_state:
                self.policy[s][a] = 0

    def initialize_eligibility_function(self):
        self.eligibility = copy.deepcopy(self.policy)
        # set all action values in each state to 0
        for state in self.eligibility:
            self.eligibility[state] = dict.fromkeys(self.eligibility[state], 0)
    '''

    def set_eligibility(self, state, action, value=None):
        if value is None:
            self.eligibility[state][action] *= (
                discount_factor_actor * eligibility_decay_actor
            )
        else:
            self.eligibility[state][action] = value

    def set_policy(self, state, action):
        self.policy[state][action] += (
            lr_actor * self.critic.get_TD_error() *
            self.eligibility[state][action]
        )
