import random

from parameters import (
    critic_type,
    episodes,
    epsilon,
    display_variable
)
from rl_actor import Actor
from rl_critic_nn import CriticNN
from rl_critic_table import CriticTable
from simworld import SimWorld

random.seed(1)


class Critic(
    CriticTable if critic_type == "table"
    else (CriticNN if critic_type == "NN"
    else False)):
    pass


class RLSystem:
    def __init__(self):
        self.sim_world = SimWorld()
        self.critic = Critic()
        self.actor = Actor()

    def get_num_input_nodes(self):
        state = self.sim_world.get_state(critic_type)
        if isinstance(state, tuple):
            return len(state)
        return 1

    def actor_critic_algorithm(self):
        if critic_type == "NN":
            input_nodes = self.get_num_input_nodes()
            self.critic.set_num_input_nodes(input_nodes)
            self.critic.build_model()

        # Enable linear decay
        eps = epsilon
        for i in range(1, episodes + 1):
            print("--- Episode: " + str(i) + " ---")

            # Reset eligibilities in actor and critic
            if critic_type == "table":
                self.critic.reset_eligibilities()
            self.actor.reset_eligibilities()

            # Get S_init and its policy
            state = self.sim_world.get_state(critic_type)
            possible_actions = self.sim_world.get_possible_actions_from_state(state)
            action = self.actor.get_best_action(state, eps, possible_actions)

            # Initialize eligibility, policy and value func
            # for start state in actor and critic_table
            if critic_type == "table":
                self.critic.add_state(state)

            self.actor.add_state(state, possible_actions)

            # Play the game
            game_over = self.sim_world.is_game_over()
            while not game_over:
                # Do action
                self.sim_world.do_action(action)
                reward = self.sim_world.get_reward()

                new_state = self.sim_world.get_state(critic_type)
                new_possible_actions = self.sim_world.get_possible_actions_from_state(new_state)

                # Add state to eligibility, policy and value funcs
                # in actor and critic_table
                if critic_type == "table":
                    self.critic.add_state(new_state)
                self.actor.add_state(new_state, new_possible_actions)

                # Check game status after new state
                game_over = self.sim_world.is_game_over()
                if not game_over:
                    # Get new action
                    new_action = self.actor.get_best_action(new_state, eps, new_possible_actions)
                else:
                    new_action = None
                # Update actor's eligibility table
                self.actor.set_eligibility(state, action, 1)

                # Update TD error
                time_out = self.sim_world.is_time_out()
                self.critic.set_TD_error(reward, state, new_state, game_over, time_out)
                TD_error = self.critic.get_TD_error()

                # Update critic's eligibility table
                if critic_type == "table":
                    self.critic.set_eligibility(state, 1)

                if critic_type == "NN":
                    self.critic.train_model(reward, state, new_state)

                for s in self.actor.eligibility.keys():
                    if critic_type == "table":
                        self.critic.set_value_for_state(s)
                        self.critic.set_eligibility(s)
                    for a in self.actor.policy[s]:
                        self.actor.set_policy(s, a, TD_error)
                        self.actor.set_eligibility(s, a)

                state = new_state
                action = new_action

            self.sim_world.save_history(i, self.actor.eligibility.keys())
            self.sim_world.reset_sim_world()

            if eps > 0.1:
                eps -= 1 / episodes

            if i > 0.95 * episodes:
                eps = 0

            if i == display_variable:
                self.sim_world.print_episode(i)
