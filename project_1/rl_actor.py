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

    def get_best_action(self, state, epsilon):
        # Change state type to string when state is a list
        if isinstance(state, list):
            state = str(state)

        # TODO: Random with epsilon probability
        random_action = random.choices(
            population=[False, True],
            weights=[1 - epsilon, epsilon],
            k=1,
        )[0]
        if random_action:
            return random.choice(
                self.sim_world.get_possible_actions_from_state(
                    state)
            )
        else:
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
            else:
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
            all_actions_for_state = self.sim_world.get_possible_actions_from_state(
                state)
            for a in all_actions_for_state:
                self.eligibility[state][a] = 0

        # For state not in eligibility
        elif state not in list(self.eligibility.keys()):
            self.eligibility[state] = dict()
            all_actions_for_state = self.sim_world.get_possible_actions_from_state(
                state)

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
            self.eligibility[state][action] *= round((
                discount_factor_actor * eligibility_decay_actor
            ), 3)
        else:
            self.eligibility[state][action] = value

    def reset_eligibilities(self):
        self.eligibility.clear()

    def set_policy(self, state, action):
        # Change state type to string when state is a list
        if isinstance(state, list):
            state = str(state)

        self.policy[state][action] += round((
            lr_actor * self.critic.get_TD_error() *
            self.eligibility[state][action]
        ), 3)
