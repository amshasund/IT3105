from parameters import pole_mass, pole_length, gravity, timestep
import random
import numpy as np


class Cart():
    def __init__(self):
        self.mass = 1
        self.location = 0
        self.velocity = 0
        self.acceleration = 0
        self.max_location = 2.4


class Pole():
    def __init__(self):
        self.mass = pole_mass
        self.length = pole_length
        self.max_angle = 0.21
        self.angle = random.randrange(-self.max_angle, self.max_angle, 0.01)
        self.angle_d = 0
        self.angle_dd = 0


class PoleBalancing():
    def __init__(self):
        self.pole = Pole
        self.cart = Cart
        self.g = gravity
        self.tau = timestep
        self.T = 300
        self.force = 10

    def add_force(self, B):
        self.pole.angle_dd = self.calculate_angle_dd(B)

    def calculate_angle_dd(self, B):
        g = self.g
        theta = self.pole.angle
        dd_theta = self.pole.angle_dd
        m_p = self.pole.mass
        L = self.pole.length
        m_c = self.cart.mass
        return (g*np.sin(theta)+(np.cos(theta)
                                 * (-B-m_p*L*dd_theta*np.sin(theta))
                                 )
                )/()
