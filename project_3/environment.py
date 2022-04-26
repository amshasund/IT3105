import numpy as np
import matplotlib.pyplot as plt
from coarse_coder import Coarse_Coder

class Acrobat:

    # STATIC PARAMETERS
    length = 1
    mass = 1
    length_cm = 0.5
    gravity = 9.8
    timestep = 0.05
    max_vel_1 = 5 #4*np.pi
    max_vel_2 = 5 #9*np.pi
    goal_height = 1 # FIXME: Need to change this
    num_actions = 3 # -1, 0, 1
    
    def __init__(self, num_bins=6):
        self.positions = [0, 0, 0, -1, 0, -2]  # [x_1, y_1, x_2, y_2, x_tip, y_tip]
        self.continuous_state = None  # [Angle 1, Angle 2, Angle 1 velocity, Angle 2 velocity]
        self.num_states = num_bins**4
        
        self.C = Coarse_Coder(num_bins, 4)
        
        self.reset()
        
        
    def reset(self):
        self.continuous_state = np.zeros(4)
        return self.get_discrete_state()
    
    
    def get_state(self):
        """Returns zeros ones string"""
        return self.C.get_state(self.continuous_state)
    
    
    def get_discrete_state(self):
        """Return a unique number for every possible state
           e.g. 3**0 + 2**1 + 2**2 + 3**3"""
        return self.C.get_discrete_state(self.continuous_state)
    
    
    def is_goal_reached(self, past_steps):
        return np.mean(past_steps[-5:]) < 400


    def step(self, action, step):  
        """Move environment to the new state"""      
        force = action-1
        theta_1, theta_2, theta_1_dot, theta_2_dot = self.continuous_state
        x_1, y_1, x_2, y_2, x_tip, y_tip = self.positions

        # Equations
        phi_2 = self.mass * self.length_cm * self.gravity * np.cos(theta_1 + theta_2 - np.pi/2)
        phi_1 = - self.mass * self.length * self.length_cm * theta_2_dot**2 * np.sin(theta_2) - 2 * (
                self.mass * self.length * self.length_cm * theta_2_dot * theta_1_dot * np.sin(theta_2)) + (
                self.mass * self.length_cm + self.mass * self.length) * self.gravity * np.cos(theta_1 - np.pi/2) + (
                phi_2) 
        delta_2 = self.mass * (self.length_cm**2 + self.length * self.length_cm * np.cos(theta_2)) + 1
        delta_1 = self.mass * self.length_cm**2 + self.mass * (
                  self.length**2 + self.length_cm**2 + 2 * self.length * self.length_cm * np.cos(theta_2)) + 2
        theta_2_ddot = (self.mass * self.length_cm**2 + 1 - delta_2**2/delta_1)**(-1) * (
                        force + (delta_2/delta_1) * phi_1 - self.mass * self.length * self.length_cm * theta_1_dot**2 * np.sin(theta_2) - phi_2)
        theta_1_ddot = - (delta_2*theta_2_ddot + phi_1)/(delta_1)
        
        # Calculate the new state
        theta_2_dot = theta_2_dot + self.timestep * theta_2_ddot
        theta_1_dot = theta_1_dot + self.timestep * theta_1_ddot
        theta_2 = theta_2 + self.timestep * theta_2_dot
        theta_1 = theta_1 + self.timestep * theta_1_dot
        
        # Position of second joint
        x_2 = x_1 + self.length * np.sin(theta_1)
        y_2 = y_1 - self.length * np.cos(theta_1)
        
        # Position of tip
        x_tip = x_2 + self.length * np.sin(theta_1 + theta_2)
        y_tip = y_2 - self.length * np.cos(theta_1 + theta_2)
        

        self.positions = [x_1, y_1, x_2, y_2, x_tip, y_tip]
        self.continuous_state = (theta_1, theta_2, theta_1_dot, theta_2_dot)
        
        # Check if bottom point is above line
        done = True if y_tip > self.goal_height else False
        
        # Create reward scheme
        if done:
            reward = 0
        else:
            reward = -1
        
        return self.get_state(), self.get_discrete_state(), reward, done
    
    
    def render(self, fig, ax, action, num_moves, num_actions):
        x_1, y_1, x_2, y_2, x_tip, y_tip = self.positions
        ax.clear()
        ax.set_xlim([0, 6])
        ax.set_ylim([-4, 2])

        ax.axhline(0)
        ax.axhline(1, ls='--')
        
        if action == 0:
            ax.axvline(1)
        elif action == 2:
            ax.axvline(5)
        
        ax.plot([x_1+3, x_2+3], [y_1, y_2]) # Anchor to first joint
        ax.plot([x_2+3, x_tip+3], [y_2, y_tip]) # First joint to second joint line
        
        ax.plot(x_1+3, y_1, marker="o", markersize=10, markeredgecolor="red", markerfacecolor="blue")
        ax.plot(x_2+3, y_2, marker="o", markersize=10, markeredgecolor="red", markerfacecolor="blue")
        ax.plot(x_tip+3, y_tip, marker="o", markersize=10, markeredgecolor="red", markerfacecolor="blue")
        
        ax.text(0.1, 1.2, f'Moves: {num_moves}', color="red", fontsize=10)
        ax.text(0.1, 1.5, f'Actions: {num_actions}', color="red", fontsize=10)
