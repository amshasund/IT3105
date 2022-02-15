import random

import tensorflow as tf

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

    # TODO: Send in possible_actions instead of using sim_world
    def get_best_action(self, state, epsilon):
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

                for values in state_actions:
                    print("b", values)
                    print("a", int(tf.reduce_max(values)))

                highest_value = tf.reduce_max(state_actions.values())
                best_actions = []
                for key, value in state_actions.items():
                    if int(value) == highest_value:
                        print("Int(value):" + str(int(value)))
                        best_actions.append(key)
                return random.choice(best_actions)

    # TODO: Send in possible_actions instead of using sim_world
    def add_state(self, state):

        # Add state to eligibility
        # For empty eligibility or  for state not in eligibility
        if state not in self.eligibility:
            self.eligibility[state] = dict()
            all_actions_for_state = self.sim_world.get_possible_actions_from_state(
                state)
            for a in all_actions_for_state:
                self.eligibility[state][a] = 0

        # Add state to policy
        # For empty policy or state not in policy
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

    # TODO: send in TD_error instead of importing critic
    def set_policy(self, state, action):

        self.policy[state][action] += (
                lr_actor * self.critic.get_TD_error() *
                self.eligibility[state][action]
        )
