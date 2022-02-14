from parameters import (
    critic_type,
    episodes,
    epsilon
)
from rl_actor import Actor
from rl_critic_nn import CriticNN
from rl_critic_table import CriticTable
from simworld import SimWorld


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
        eps = epsilon

        for i in range(1, episodes + 1):
            print("--- Episode: " + str(i) + " ---")
            # Reset eligibilities in actor and critic
            if critic_type == "table":
                self.critic.reset_eligibilities()
            self.actor.reset_eligibilities()
            # Get S_init and its policy
            # TODO: critic or sim_world?
            state = self.critic.get_state()
            action = self.actor.get_best_action(state, eps)

            # Initialize eligibility, policy and value func
            # for start state in actor and critic_table
            if critic_type == "table":
                self.critic.add_state(state)

            self.actor.add_state(state)

            # Play the game
            game_over = self.sim_world.is_game_over()
            while not game_over:
                # Do action
                self.sim_world.do_action(action)
                reward = self.sim_world.get_reward()

                # TODO: change to critic.get_state?
                new_state = self.sim_world.get_state()
                # print("New State: " + str(new_state))

                # Add state to eligibility, policy and value funcs
                # in actor and critic_table
                if critic_type == "table":
                    self.critic.add_state(new_state)
                self.actor.add_state(new_state)

                # TODO: Check this - does it save right
                self.sim_world.save_history(i)

                # Check game status after new state
                game_over = self.sim_world.is_game_over()
                if not game_over:
                    # Get new action
                    new_action = self.actor.get_best_action(new_state, eps)
                else:
                    new_action = None
                # Update actor's eligibility table
                self.actor.set_eligibility(state, action, 1)

                # Update TD error
                self.critic.set_TD_error(reward, state, new_state)

                # Update critic's eligibility table
                if critic_type == "table":
                    self.critic.set_eligibility(state, 1)

                # TODO: training before forloop?
                # TODO: do this after the episode with a list of all the states
                # Should we still loop for actor?
                if critic_type == "NN":
                    # fix state input format
                    self.critic.train_model(reward, state, new_state)

                for s in self.actor.eligibility.keys():
                    if critic_type == "table":
                        self.critic.set_value_for_state(s)
                        self.critic.set_eligibility(s)
                    for a in self.actor.policy[s]:
                        self.actor.set_policy(s, a)
                        self.actor.set_eligibility(s, a)

                state = new_state
                action = new_action

            self.sim_world.reset_sim_world()

            if eps > 0.1:
                eps -= 1/episodes
