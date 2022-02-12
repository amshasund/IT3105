import random

import matplotlib.pyplot as plt
import numpy as np

from parameters import pole_mass, pole_length, gravity, timestep


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
        return random.randrange(-self.max_angle, self.max_angle, 0.01)

    def reset_pole(self):
        self.angle = self.set_start_angle()
        self.angular_velocity = 0
        self.angular_acceleration = 0


class PolePlayer:
    def __init__(self, env):
        self.env = env
        self.situation = self.update_situation()
        self.reward = 0

    def get_situation(self):
        return self.situation

    def get_reward(self):
        pass

    def add_force(self, action):
        self.env.update_state(action)
        self.update_situation()

    def update_situation(self):
        sit = [
            self.env.cart.location,
            self.env.cart.velocity,
            self.env.cart.acceleration,
            self.env.pole.angle,
            self.env.pole.angular_velocity,
            self.env.pole.angular_acceleration
        ]
        return sit

    def set_start_situation(self):
        self.env.reset_environment()
        self.situation = self.update_situation()

    def get_legal_push(self):
        return self.env.get_force_options()


class PoleEnv:
    def __init__(self):
        self.pole = Pole()
        self.cart = Cart()
        self.g = gravity
        self.tau = timestep
        self.T = 300
        self.force = 10

    def reset_environment(self):
        self.pole.reset_pole()
        self.cart.reset_cart()

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

    # def get_state(self):

    # def get_reward(self):

    def update_angular_acceleration(self, B):
        g = self.g
        theta = self.pole.angle
        dd_theta = self.pole.angular_acceleration
        m_p = self.pole.mass
        L = self.pole.length
        m_c = self.cart.mass

        return (
            g * np.sin(theta)
            + (np.cos(theta) * (-B - m_p * L * dd_theta * np.sin(theta))) / (m_p + m_c)
        ) / (L * ((4 / 3) - (m_p * np.cos(theta) ** 2) / (m_p + m_c)))

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
                print("Keep on playing!")
                return True
            print("Pole out of range")
        print("Cart out of range")
        return False


class PoleWorld:
    def __init__(self):
        self.environment = PoleEnv()
        self.player = PolePlayer(self.environment)

    def get_actions(self):
        return self.player.get_legal_push()

    def get_state(self):
        return self.player.get_units()

    def get_all_possible_states(self):
        return self.environment.get_range_of_units()

    def get_possible_actions_from_state(self, state):
        # in this game, state equals amount of units
        return self.environment.get_legal_bets(state)

    def do_action(self, action):
        self.player.place_bet(action)

    def get_reward(self):
        return self.player.get_reward()

    def is_game_over(self):
        state = self.get_state()
        if state == 100:
            # print("You won!")
            # Reset number of units for new game
            self.player.set_start_situation()
            return True
        elif state == 0:
            # print("You lost..")
            # Reset number of units for new game
            self.player.set_start_situation()
            return True
        return False

    @staticmethod
    def print_results(policy):
        states = list(policy.keys())
        bets = list(policy.values())
        best_bets = [0]

        for dict in bets[1:100]:
            best_bets.append(max(dict, key=dict.get))

        best_bets.append(0)
        # print("Best_bets: " + str(best_bets))

        # Plotting the points
        plt.plot(states, best_bets)

        # Name the axis and set title
        plt.xlabel("State")
        plt.ylabel("Bet")
        plt.title("Policy after episodes")
        plt.show()
