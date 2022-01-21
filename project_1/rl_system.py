from parameters import critic_type
from simworld import SimWorld
import random


class Actor:
    def __init__(self, critic):
        self.critic = critic


class Critic:
    def __init__(self, type):
        self.type = type
        self.V = dict()

    def initialize_value_function(self, all_states):
        for s in all_states:
            self.V[s] = random.randint(0, 10)


class RLSystem:
    def __init__(self):
        self.critic = Critic(critic_type)
        self.actor = Actor(self.critic)
        self.sim_world = SimWorld()

    def actor_critic_algorithm(self):
        # Initialize
        all_states = self.sim_world.get_all_possible_states()
        self.critic.initialize_value_function(all_states)


