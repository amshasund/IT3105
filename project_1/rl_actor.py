import random

from parameters import (
    lr_actor,
    discount_factor_actor,
    eligibility_decay_actor
)


class Actor:
    def __init__(self):
        self.policy = dict()
        self.eligibility = dict()

    def get_best_action(self, state, epsilon, possible_actions):
        random_action = random.choices(
            population=[False, True],
            weights=[1 - epsilon, epsilon],
            k=1,
        )[0]
        # For random choice, policy being empty or for state NOT in policy
        if (random_action) or (state not in self.policy):
            return random.choice(possible_actions)
        # Choosing action based on experience
        else:
            state_actions = self.policy[state]
            highest_value = max(state_actions.values())
            best_actions = []

            # Choose random from the best actions
            for key, value in state_actions.items():
                if value == highest_value:
                    best_actions.append(key)
            return random.choice(best_actions)

    def add_state(self, state, possible_actions):
        """ Method that initializes a state in eligibility and policy """
        # Add state to eligibility
        if state not in self.eligibility:
            self.eligibility[state] = dict()
            for a in possible_actions:
                self.eligibility[state][a] = 0

        # Add state to policy
        if state not in self.policy:
            self.policy[state] = dict.fromkeys(self.eligibility[state], 0)

    def set_eligibility(self, state, action, value=None):

        if value is None:
            self.eligibility[state][action] *= (
                    discount_factor_actor * eligibility_decay_actor
            )
        else:
            self.eligibility[state][action] = value

    def reset_eligibilities(self):
        self.eligibility.clear()

    def set_policy(self, state, action, TD_error):
        self.policy[state][action] += (
                lr_actor * TD_error *
                self.eligibility[state][action]
        )
