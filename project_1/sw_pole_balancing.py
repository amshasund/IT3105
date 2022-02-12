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


class Pole:
    def __init__(self):
        self.mass = pole_mass
        self.length = pole_length
        self.max_angle = 0.21  # in both directions
        self.angle = random.randrange(-self.max_angle, self.max_angle, 0.01)
        self.angular_velocity = 0
        self.angular_acceleration = 0


class PolePlayer:
    def __init__(self, env):
        self.env = env
        self.reward = 0
        self.B = 0

    def add_force(self, action):
        if action == 0:
            self.B = -1
        elif action == 1:
            self.B = 1
        self.env.add_force(self.B)

    def get_angle(self):
        return self.units

    def get_reward(self):
        # For winning
        if self.units == 100:
            self.reward = 100

        # For loosing
        elif self.units == 0:
            self.reward = -100

        # For moving
        else:
            self.reward = (
                -1
            )  # økte antall minus per steg # sjekke antall steg økte antall episoder

        return self.reward

    def set_start_units(self):
        self.units = random.randint(1, 99)

    def place_bet(self, bet):
        # self.reward = self.env.perform_bet(bet)
        self.units += self.env.perform_bet(bet)

    def get_possible_forces(self):
        return self.env.get_legal_bets(self.units)


class PoleEnv:
    def __init__(self):
        self.pole = Pole()
        self.cart = Cart()
        self.g = gravity
        self.tau = timestep
        self.T = 300
        self.force = 10

    def add_force(self, B):
        if B == -1:
            B = -self.force
        elif B == 1:
            B = self.force
        else:
            return

        self.pole.angular_acceleration = self.update_angular_acceleration(B)
        self.cart.acceleration = self.update_acceleration(B)

        self.pole.angular_velocity = (
                self.pole.angular_velocity + self.tau * self.pole.angular_acceleration
        )
        self.cart.velocity = self.cart.velocity + self.tau * self.cart.acceleration
        self.pole.angle = self.pole.angle + self.tau * self.pole.angular_velocity
        self.cart.location = self.cart.location + self.tau * self.cart.velocity
        return self.is_successfull()

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
                       B + m_p * L * (d_theta ** 2 * np.sin(theta) - dd_theta * np.cos(theta))
               ) / (m_p + m_c)

    def is_successfull(self):
        if -self.cart.max_location <= self.cart.location <= self.cart.max_location:
            if -self.pole.max_angle <= self.pole.angle <= self.pole.max_angle:
                print("Keep on playing!")
                return True
            print("Pole out of range")
        print("Cart out of range")

        print("You were unsuccessfull.")
        return False

    def game_over(self):
        if self.is_successfull == False:
            print("GAME OVER.")
            return True
        return False


class PoleWorld:
    def __init__(self):
        self.environment = PoleEnv()
        self.player = PolePlayer(self.environment)

    def get_actions(self):
        return self.player.get_possible_forces()

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
            self.player.set_start_units()
            return True
        elif state == 0:
            # print("You lost..")
            # Reset number of units for new game
            self.player.set_start_units()
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
