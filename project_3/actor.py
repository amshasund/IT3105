import numpy as np


class Actor:
    
    def __init__(self, num_actions, num_states):
        self.num_actions = num_actions
        self.num_states = num_states
        self.learning_rate = 0.01
        self.expl_rate = 0.6
        self.expl_decay = self.expl_rate/800
        self.discount_factor = 0.96
        self.trace_decay = 0.96
        
        self.table = np.zeros(shape=(num_states, num_actions))

        self.eligibilities = np.zeros(shape=(num_states, num_actions))
        
        
        
    def prepare_for_epoch(self):
        self.expl_rate -= self.expl_decay
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