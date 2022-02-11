import matplotlib.pyplot as plt

from parameters import (
    critic_type,
    episodes
)
from simworld import SimWorld

from rl_critic_nn import CriticNN
from rl_critic_table import CriticTable
from rl_actor import Actor


class Critic(
        CriticTable if critic_type == "table"
        else (CriticNN if critic_type == "NN"
              else False)):
    pass


class RLSystem:
    def __init__(self):
        self.sim_world = SimWorld()
        self.critic = Critic(self.sim_world)
        self.actor = Actor(self.critic, self.sim_world)

    def actor_critic_algorithm(self):
        # Initialize
        all_states = self.sim_world.get_all_possible_states()
        if critic_type == "table":
            self.critic.initialize_value_function(all_states)
        self.actor.initialize_policy_function(all_states)
        acc_reward = [0] * episodes

        for i in range(1, episodes + 1):
            # Reset eligibilities in actor and critic
            self.actor.initialize_eligibility_function()
            if critic_type == "table":
                self.critic.initialize_eligibility_function(all_states)

            # Get S_init and its policy
            # TODO: critic or sim_world?
            state = self.critic.get_state()
            #print("Your start state: " + str(state))
            action = self.actor.get_best_action(state)

            # Play the game
            game_over = self.sim_world.is_game_over()
            while not game_over:
                # Do action
                self.sim_world.do_action(action)
                reward = self.sim_world.get_reward()
                acc_reward[i - 1] += reward

                # TODO: change to critic.get_state?
                new_state = self.sim_world.get_state()
                #print("New State: " + str(new_state))

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
                if critic_type == "table":
                    self.critic.set_eligibility(state, 1)

                # TODO: training before forloop?
                # Should we still loop for actor?
                if critic_type == "NN":
                    self.critic.train_model(reward, state, new_state)
                for s in all_states:
                    if critic_type == "table":
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
