import numpy as np
import matplotlib.pyplot as plt


class Actor:
    
    def __init__(self, num_actions, num_states, learning_rate=0.05, discount_factor=0.95, expl_rate=0.5, expl_decay=0.99, trace_decay=0.95):
        self.num_actions = num_actions
        self.num_states = num_states
        self.learning_rate = learning_rate
        self.expl_rate = expl_rate
        self.expl_decay = expl_decay
        self.discount_factor = discount_factor
        self.trace_decay = trace_decay
        
        self.table = np.zeros(shape=(num_states, num_actions))

        self.eligibilities = np.zeros(shape=(num_states, num_actions))
        
        
        
    def prepare_for_epoch(self):
        self.expl_rate = (self.expl_rate-0.005)*self.expl_decay + 0.005
        self.eligibilities = np.zeros(shape=(self.num_states, self.num_actions))
        
        
    def get_action(self, state):
        enable_exploration = np.random.uniform(0, 1) < self.expl_rate
        if enable_exploration:
            return np.random.randint(0, self.num_actions)
        else:
            return np.argmax(self.table[state])
    
    
    def trace_eligibilities(self, delta, state, action):
        self.eligibilities[state, action] = 1.0
        self.table += self.learning_rate*delta*self.eligibilities
        self.eligibilities *= self.discount_factor*self.trace_decay