from parameters import critic_type, episodes
from simworld import SimWorld
import random
import copy


class Actor:
    def __init__(self, critic, sim_world):
        self.critic = critic
        self.sim_world = sim_world
        self.policy = dict()
        self.eligibility = dict()

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


class Critic:
    def __init__(self, type, sim_world):
        self.type = type
        self.sim_world = sim_world
        self.V = dict()
        self.eligibility = dict()

    def initialize_value_function(self, all_states):
        for s in all_states:
            self.V[s] = random.randint(0, 10)

    # should this be done differently?
    def initialize_eligibility_function(self, all_states):
        for s in all_states:
            self.eligibility[s] = 0


class RLSystem:
    def __init__(self):
        self.sim_world = SimWorld()
        self.critic = Critic(critic_type, self.sim_world)
        self.actor = Actor(self.critic, self.sim_world)

    def actor_critic_algorithm(self):
        # Initialize
        all_states = self.sim_world.get_all_possible_states()
        self.critic.initialize_value_function(all_states)
        self.actor.initialize_policy_function(all_states)

        for e in episodes:
            # Reset eligibilities in actor and critic
            self.actor.initialize_eligibility_function()
            self.critic.initialize_eligibility_function(all_states)
