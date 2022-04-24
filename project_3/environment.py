import numpy as np
import matplotlib.pyplot as plt

class Acrobat:

    # STATIC PARAMETERS
    length = 1
    mass = 1
    length_cm = 0.5
    gravity = 9.8
    timestep = 0.05
    max_vel = 5 # FIXME: Need to change this
    max_force = 1
    goal_height = 1 # FIXME: Need to change this
    num_actions = 3
    
    def __init__(self, num_bins=7): 
        # [x_1, y_1, x_2, y_2, x_tip, y_tip]
        self.positions = [0, 0, 0, -1, 0, -2]
        # [Angle 1, Angle 2, Angle 1 velocity, Angle 2 velocity]
        self.continuous_state = None
        self.state_bins = [
            # Angle 1
            np.linspace(-np.pi, np.pi, num_bins + 1)[1:-1],
            # Angle 2
            np.linspace(-np.pi, np.pi, num_bins + 1)[1:-1],
            # Angle 1 velocity
            np.linspace(-self.max_vel, self.max_vel, num_bins + 1)[1:-1],
            # Angle 2 velocity
            np.linspace(-self.max_vel, self.max_vel, num_bins + 1)[1:-1]
            ]
        self.num_bins = max(len(b) for b in self.state_bins) + 1
        self.num_states = self.num_bins**len(self.state_bins)

        self.reset()
        
        
    def reset(self):
        self.continuous_state = np.zeros(4)
        return self.get_discrete_state()


    def get_state(self):
        """Returns zeros and ones string"""
        out = np.zeros((4, self.num_bins))
        out[np.arange(4), self.digitize().reshape(-1)] = 1.
        return np.array(out).reshape(-1)


    def digitize(self):
        """Returns quadruple e.g. array([3, 2, 2, 3]) 
           number of bin to which given variable belongs"""
        return np.array([np.digitize(self.continuous_state[0], self.state_bins[0]),
                         np.digitize(self.continuous_state[1], self.state_bins[1]),
                         np.digitize(self.continuous_state[2], self.state_bins[2]),
                         np.digitize(self.continuous_state[3], self.state_bins[3])])


    
    def get_discrete_state(self):
        """Return a unique number for every possible state
           e.g. 3**0 + 2**1 + 2**2 + 3**3"""
        return np.dot(self.digitize(), self.num_bins**np.arange(4))


    
    def step(self, action, step, run):  
        """Move environment to the new state"""      
        force = action-1
        theta_1, theta_2, theta_1_dot, theta_2_dot = self.continuous_state
        x_1, y_1, x_2, y_2, x_tip, y_tip = self.positions

        # Equations
        phi_2 = self.mass * self.length_cm * np.cos(theta_1 + theta_2 - np.pi/2)
        phi_1 = - self.mass * self.length * self.length_cm * theta_2_dot**2 * np.sin(theta_2) - 2 * (
                self.mass * self.length * self.length_cm * theta_2_dot * theta_1_dot * np.sin(theta_2)) + (
                self.mass * self.length_cm + self.mass * self.length) * self.gravity * np.cos(theta_1 - np.pi/2) + (
                phi_2) 
        delta_2 = self.mass * (self.length_cm** + self.length * self.length_cm * np.cos(theta_2)) + 1
        delta_1 = self.mass * self.length_cm**2 + self.mass * (
                  self.length**1 + self.length_cm**2 + 2 * self.length * self.length_cm * np.cos(theta_2)) + 2
        theta_2_ddot = (self.mass * self.length_cm**2 + 1 - delta_2**2/delta_1)**(-1) * (
                        force + delta_2/delta_1 * phi_1 - self.mass * self.length * self.length_cm * theta_1_dot**2 * np.sin(theta_2) - phi_2)
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
        
        # TODO: Check if bottom point is above line
        done = True if y_tip > self.goal_height else False
        
        # TODO: Create reward scheme
        if done:
            reward = 10
        else:
            reward = -0.1
        
        return self.get_state(), self.get_discrete_state(), reward, done
    
    
    def create_figure(self):
        fig, ax = plt.subplots(1, figsize=(16, 8))
        return fig, ax
    
    
    def render(self, fig, ax):
        x_1, y_1, x_2, y_2, x_tip, y_tip = self.positions
        ax.clear()
        ax.set_xlim([0, 10])
        ax.set_ylim([-5, 1])

        ax.axhline(0)
        
        ax.plot([x_1+5, x_2+5], [y_1, y_2]) # Anchor to first joint

        ax.plot([x_2+5, x_tip+5], [y_2, y_tip]) # First joint to second joint line
        
        ax.plot(x_1+5, y_1, marker="o", markersize=10, markeredgecolor="red", markerfacecolor="blue")
        ax.plot(x_2+5, y_2, marker="o", markersize=10, markeredgecolor="red", markerfacecolor="blue")
        ax.plot(x_tip+5, y_tip, marker="o", markersize=10, markeredgecolor="red", markerfacecolor="blue")
