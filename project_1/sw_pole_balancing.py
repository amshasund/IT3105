import os
import sys
import inspect

# to get parameters from parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from parameters import pole_mass, pole_length, gravity, timestep
import random
import numpy as np


class PoleWorld:
    def __init__(self):
        pass


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


class PoleBalancingPlayer:
    def __init__(self):
        self.B = 0

        self.prev_state = 0
        self.action = 0
        self.current_state = 0
        self.reward = 0

        self.possible_actions = [0, 1]
        self.possible_states = [(), ()]
        self.possible_rewards = [0, 0]

        self.world = PoleBalancingEnvironment

    def do_action(self, action):
        if action == 0:
            self.B = -1
        elif action == 1:
            self.B = 1
        self.world.add_force(self.B)


class PoleBalancingEnvironment:
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
            B + m_p * L * (d_theta**2 * np.sin(theta) - dd_theta * np.cos(theta))
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
