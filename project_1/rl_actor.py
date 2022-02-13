import random

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
        # Change state type to string when state is a list
        if isinstance(state, list):
            state = str(state)

        # For policy being empty
        if not self.policy:
            return random.choice(
                self.sim_world.get_possible_actions_from_state(
                    state)
            )

        # For state NOT in policy
        elif state not in list(self.policy.keys()):
            return random.choice(
                self.sim_world.get_possible_actions_from_state(
                    state)
            )

        # For state in policy
        else:
            # print("state in policy")
            state_actions = self.policy[state]
            # TODO: shorten this?
            highest_value = max(state_actions.values())
            best_actions = []
            for key, value in state_actions.items():
                if value == highest_value:
                    best_actions.append(key)
            return random.choice(best_actions)

    def add_state(self, state):
        # Change state type to string when state is a list
        if isinstance(state, list):
            state = str(state)

        # Add state to eligibility
        # For empty eligibility
        if not self.eligibility:
            self.eligibility[state] = dict()
            all_actions_for_state = \
                self.sim_world.get_possible_actions_from_state(state)
            for a in all_actions_for_state:
                self.eligibility[state][a] = 0

        # For state not in eligibility
        elif state not in list(self.eligibility.keys()):
            self.eligibility[state] = dict()
            all_actions_for_state = \
                self.sim_world.get_possible_actions_from_state(state)

            for a in all_actions_for_state:
                self.eligibility[state][a] = 0

        # Add state to policy
        # For empty policy
        if not self.policy:
            self.policy[state] = dict.fromkeys(self.eligibility[state], 0)

        # For state not in policy
        elif state not in list(self.policy.keys()):
            self.policy[state] = dict.fromkeys(self.eligibility[state], 0)

    def set_eligibility(self, state, action, value=None):
        # Change state type to string when state is a list
        if isinstance(state, list):
            state = str(state)

        if value is None:
            self.eligibility[state][action] *= (
                    discount_factor_actor * eligibility_decay_actor
            )
        else:
            self.eligibility[state][action] = value

    def set_policy(self, state, action):
        # Change state type to string when state is a list
        if isinstance(state, list):
            state = str(state)

        self.policy[state][action] += (
                lr_actor * self.critic.get_TD_error() *
                self.eligibility[state][action]
        )
