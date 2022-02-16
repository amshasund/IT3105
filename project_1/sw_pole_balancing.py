import copy
import random

import matplotlib.pyplot as plt
import numpy as np

from parameters import pole_mass, pole_length, gravity, timestep, episodes, max_steps


class Cart:
    def __init__(self):
        self.mass = 1
        self.location = 0
        self.velocity = 0
        self.acceleration = 0
        self.max_location = 2.4  # in both directions

    def reset_cart(self):
        self.location = 0
        self.velocity = 0
        self.acceleration = 0


class Pole:
    def __init__(self):
        self.mass = pole_mass
        self.length = pole_length
        self.max_angle = 0.21  # in both directions
        self.angle = self.set_start_angle()
        self.angular_velocity = 0
        self.angular_acceleration = 0

    def set_start_angle(self):
        return round(random.uniform(-self.max_angle, self.max_angle), 2)

    def reset_pole(self):
        self.angle = self.set_start_angle()
        self.angular_velocity = 0
        self.angular_acceleration = 0


class PolePlayer:
    def __init__(self, env):
        self.env = env
        self.situation = self.update_situation()
        self.reward = 0
        self.num_pushes = 0

    def get_situation(self):
        self.situation = self.update_situation()
        return self.situation

    def get_reward(self):
        # For loosing
        if not self.env.is_state_legal():
            self.reward = -5

        # For winning
        if self.num_pushes == 100:
            self.reward = 2
        elif self.num_pushes == 200:
            self.reward = 3
        elif self.num_pushes == max_steps:
            self.reward = 5

        # For legal moving
        else:
            self.reward = 1  # 1 * self.num_pushes // 10

        return self.reward

    def update_situation(self):
        # TODO: Try with booleans
        location = True if self.env.cart.location > 0 else False
        angle = True if self.env.pole.angle > 0 else False
        angle_vel = True if self.env.pole.angular_velocity > 0 else False
        location_vel = True if self.env.cart.velocity > 0 else False

        situation = [location,
                     location_vel,
                     angle,
                     angle_vel
                     ]

        # situation = [
        #   round(self.env.cart.location, 2),
        #  round(self.env.cart.velocity, 2),
        # round(self.env.pole.angle, 2),
        # round(self.env.pole.angular_velocity, 2),
        # ]
        return situation

    def add_force(self, action):
        # Add force to the environment
        self.env.update_state(action)

        # Update the players state
        self.update_situation()

        # Increase number of pushes
        self.num_pushes += 1

    def get_legal_push(self):
        return self.env.get_force_options()

    def reset_player(self):
        self.situation = self.update_situation()
        self.reward = 0
        self.num_pushes = 0


class PoleEnv:
    def __init__(self):
        self.pole = Pole()
        self.cart = Cart()
        self.g = gravity
        self.tau = timestep
        self.T = 300
        self.force = 10

    def update_state(self, bang_bang):
        self.pole.angular_acceleration = self.update_angular_acceleration(
            bang_bang)
        self.cart.acceleration = self.update_acceleration(bang_bang)

        self.pole.angular_velocity = (
                self.pole.angular_velocity + self.tau * self.pole.angular_acceleration
        )
        self.cart.velocity = self.cart.velocity + self.tau * self.cart.acceleration
        self.pole.angle = self.pole.angle + self.tau * self.pole.angular_velocity
        self.cart.location = self.cart.location + self.tau * self.cart.velocity

    def get_force_options(self):
        return [-self.force, self.force]

    def update_angular_acceleration(self, B):
        g = self.g
        theta = self.pole.angle
        d_theta = self.pole.angular_velocity
        m_p = self.pole.mass
        L = self.pole.length
        m_c = self.cart.mass

        return (
                       g * np.sin(theta)
                       + (np.cos(theta) * (-B - m_p * L *
                                           (d_theta ** 2) * np.sin(theta))) / (m_p + m_c)
               ) / (L * ((4 / 3) - (m_p * (np.cos(theta)) ** 2) / (m_p + m_c)))

    def update_acceleration(self, B):
        theta = self.pole.angle
        d_theta = self.pole.angular_velocity
        dd_theta = self.pole.angular_acceleration
        m_p = self.pole.mass
        L = self.pole.length
        m_c = self.cart.mass

        return (
                       B + m_p * L * (d_theta ** 2 * np.sin(theta) -
                                      dd_theta * np.cos(theta))
               ) / (m_p + m_c)

    def is_state_legal(self):
        if -self.cart.max_location <= self.cart.location <= self.cart.max_location:
            if -self.pole.max_angle <= self.pole.angle <= self.pole.max_angle:
                return True
        return False

    def reset_environment(self):
        self.pole.reset_pole()
        self.cart.reset_cart()


class PoleWorld:
    def __init__(self):
        self.environment = PoleEnv()
        self.player = PolePlayer(self.environment)
        self.moves_per_episode = [0] * (episodes + 1)
        self.current_angles = []
        self.best_angles = []
        self.best_episode = 0

    def get_actions(self):
        return self.player.get_legal_push()

    def get_state(self):
        state = copy.deepcopy(self.player.get_situation())
        state = tuple(state)
        return state

    def get_all_possible_states(self):
        pass

    def get_possible_actions_from_state(self, state):
        # all actions are legal for all states
        return self.get_actions()

    def do_action(self, action):
        self.player.add_force(action)
        self.save_state()

    def get_reward(self):
        return self.player.get_reward()

    def is_game_over(self):
        # Pole out of balance
        if not self.environment.is_state_legal():
            return True

        # Maximum timesteps reached
        elif self.player.num_pushes == max_steps:
            print("Max timesteps reached!")
            return True

        else:
            return False

    def reset_sim_world(self):
        self.environment.reset_environment()
        self.player.reset_player()
        self.current_angles = []

    def save_state(self):
        self.current_angles.append(self.environment.pole.angle)

    def save_history(self, episode, state=None):
        self.moves_per_episode[episode] = self.player.num_pushes
        print("Number of moves: ", self.moves_per_episode[episode])

        if self.moves_per_episode[episode] > self.moves_per_episode[self.best_episode]:
            self.best_angles = []
            self.best_episode = episode
            self.best_angles = copy.deepcopy(self.current_angles)

    def print_end_results(self, policy):
        # Plot: The Progression of Learning
        x = list(range(1, episodes + 1))
        y = self.moves_per_episode[1:]  # From episode 1 to 'episode'

        plt.plot(x, y)
        plt.xlabel("Episode")
        plt.ylabel("Timestep")
        plt.title("The Progression of Learning")
        plt.show()

    def print_episode(self, episode):
        # Plot the most successful episode
        num_moves = self.moves_per_episode[episode]
        x = list(range(num_moves))
        y = self.best_angles

        plt.plot(x, y)
        plt.xlabel("Timesteps")
        plt.ylabel("Angle (Radians)")
        plt.title("Most successfull episode nr {}".format(self.best_episode))
        plt.show()
