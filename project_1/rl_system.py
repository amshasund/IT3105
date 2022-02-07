import copy
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from parameters import (
    critic_type,
    episodes,
    lr_critic,
    lr_actor,
    discount_factor_critic,
    discount_factor_actor,
    eligibility_decay_critic,
    eligibility_decay_actor,
)
from simworld import SimWorld


class Actor:
    def __init__(self, critic, sim_world):
        self.critic = critic
        self.sim_world = sim_world
        self.policy = dict()
        self.eligibility = dict()

    def get_best_action(self, state):
        actions = self.policy[state]
        highest_value = max(actions.values())
        best_actions = []
        for key, value in actions.items():
            if value == highest_value:
                best_actions.append(key)
        return random.choice(best_actions)

    def initialize_policy_function(self, all_states):
        for s in all_states:
            self.policy[s] = dict()
            all_actions_for_state = self.sim_world.get_possible_actions_from_state(s)
            for a in all_actions_for_state:
                self.policy[s][a] = 0

    def initialize_eligibility_function(self):
        self.eligibility = copy.deepcopy(self.policy)
        # set all action values in each state to 0
        for state in self.eligibility:
            self.eligibility[state] = dict.fromkeys(self.eligibility[state], 0)

    def set_eligibility(self, state, action, value):
        if value is None:
            self.eligibility[state][action] *= (
                discount_factor_actor * eligibility_decay_actor
            )
        else:
            self.eligibility[state][action] = value

    def set_policy(self, state, action):
        self.policy[state][action] += (
            lr_actor * self.critic.get_TD_error() * self.eligibility[state][action]
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
            self.eligibility[state] *= discount_factor_critic * eligibility_decay_critic
        else:
            self.eligibility[state] = value

    def set_value_for_state(self, state):
        self.V[state] += lr_critic * self.get_TD_error() * self.eligibility[state]


class CriticANN:
    def __init__(self, sim_world):
        self.sim_world = sim_world
        self.nn_model = self.build_model()
        self.TD_error = 0

    def build_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers)

    def get_state(self):
        state = self.sim_world.get_state()
        return self.state_to_binary(state)

    def state_to_binary(self, state):
        binary_state = ""
        if isinstance(state, list):
            for element in state:
                binary_state += np.binary_repr(element, 8)
        else:
            binary_state += np.binary_repr(state, 8)
        return binary_state

    def binary_to_state(self, binary_state):
        state = []
        if len(binary_state) > 8:
            for i in range(0, len(binary_state), 8):
                state.append(self.binary_to_decimal(binary_state[i : i + 8], 8))
        else:
            state = self.binary_to_decimal(binary_state, 8)
        return state

    def binary_to_decimal(self, num, bits):
        # to handle negative numbers
        if num[0] == "1":
            return -(2**bits - int(num, 2))
        return int(num, 2)

    def get_value(self, state):
        pass


class Critic(
    CriticTable
    if critic_type == "table"
    else (CriticANN if critic_type == "ANN" else False)
):
    pass


class RLSystem:
    def __init__(self):
        self.sim_world = SimWorld()
        self.critic = Critic(self.sim_world)
        self.actor = Actor(self.critic, self.sim_world)

    def actor_critic_algorithm(self):
        # Initialize
        all_states = self.sim_world.get_all_possible_states()
        self.critic.initialize_value_function(all_states)
        self.actor.initialize_policy_function(all_states)
        acc_reward = [0] * episodes

        for i in range(1, episodes + 1):
            # print("---------- Episode nr: " + str(i) + " ----------")

            # Reset eligibilities in actor and critic
            self.actor.initialize_eligibility_function()
            self.critic.initialize_eligibility_function(all_states)

            # Get S_init and its policy
            state = self.critic.get_state()
            # print("Your start state: " + str(state))
            action = self.actor.get_best_action(state)

            # Play the game
            game_over = self.sim_world.is_game_over()
            while not game_over:
                # Do action
                self.sim_world.do_action(action)
                reward = self.sim_world.get_reward()
                acc_reward[i - 1] += reward
                new_state = self.sim_world.get_state()
                # print("New State: " + str(new_state))

                # Check game status after new state
                game_over = self.sim_world.is_game_over()
                if game_over:
                    break

                # Get new action
                new_action = self.actor.get_best_action(new_state)

                # Update actor's eligibility table
                self.actor.set_eligibility(state, action, 1)

                # Update TD error
                self.critic.set_TD_error(reward, state, new_state)

                # Update critic's eligibility table
                self.critic.set_eligibility(state, 1)

                for s in all_states:
                    self.critic.set_value_for_state(s)
                    self.critic.set_eligibility(s, None)
                    for a in self.actor.policy[s]:
                        self.actor.set_eligibility(s, a, None)
                        self.actor.set_policy(s, a)

                state = new_state
                action = new_action
        self.plot_train_progression(acc_reward)

    def plot_train_progression(self, acc_reward):
        list_episodes = list(range(episodes))
        # Plotting the points
        plt.plot(list_episodes, acc_reward)
        # Plot heller snittet for hvert 100ede episode

        # Name the axis and set title
        plt.xlabel("Episode")
        plt.ylabel("Accumulated reward")
        plt.title("Accumulated reward for each episode")
        plt.show()
