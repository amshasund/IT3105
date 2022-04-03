import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from parameters import (
    hex_board_size,
    hidden_layers,
    activation_function,
    optimizer, 
)

 
class ANet:
    def __init__(self):
        self.num_input_nodes = hex_board_size**2 + 1  
        # one extra input to know which player
        self.model = None

    def build_model(self, loss=tf.keras.losses.binary_crossentropy):
        # Create Neural Net
        model = tf.keras.models.Sequential()

        # Add input layer
        model.add(tf.keras.layers.Input(shape=(self.num_input_nodes,)))

        # Add hidden layers
        for i in range(len(hidden_layers)):
            model.add(tf.keras.layers.Dense(
                hidden_layers[i],
                activation=activation_function[i]))

        # Add output layer
        # TODO: should we remove the already placed pieces from this?
        model.add(tf.keras.layers.Dense(
            hex_board_size**2, activation='softmax'))

        # Using stochastic gradient descent when compiling
        # Optimizer is a string, could be a problem that it is not a tf function
        model.compile(optimizer=optimizer, loss=loss,
                      metrics=[
                          tf.keras.metrics.categorical_accuracy])

        self.model = model

    def train_model(self, rbuf):
        # Input: dict{  key=tuple(player, board.flatten()), 
        #               value=board's visit distribution)}
        for key in rbuf:
            # Train on random batches of rbuf
            if np.random.choice([True, False], p=[0.5, 0.5]):
                state = np.array(key, dtype=float)
                state = self.reshape_state(state)
                target = np.array(rbuf[key], dtype=float).flatten()
                
                # Normalize replay buffer data
                target = target / target.sum()
                target = self.reshape_state(target)
                
                self.model.fit(state, target, verbose=0)


    def rollout(self, state, legal_actions):
        # Make state ready for input to actor net model
        state_to_model = self.reshape_state(state)

        # Get preference distribution from actor net model
        distribution = np.array(self.model(state_to_model)[0])

        # Eliminate illegal moves
        distribution = distribution * legal_actions

        # Normalize distribution to ensure no error from np.random.choice
        
        # For cases when distribution is all zeroes 
        if (np.sum(distribution) != 0):
            distribution = distribution / (np.sum(distribution))
        else:
            distribution = legal_actions / (np.sum(legal_actions))
        
        # When choosing move, use prob from anet to choose a move [0.4, 0.45, 0.1, 0.05]
        # Find 1d index of flattened distribution
        return np.random.choice(range(len(distribution)), p=distribution)


    def save_model(self, game_nr):
        self.model.save("model34x4_{nr}.h5".format(nr=game_nr))

        '''
        # Calling `save('my_model')` creates a SavedModel folder `my_model`.
        model.save("my_model")

        # It can be used to reconstruct the model identically.
        reconstructed_model = keras.models.load_model("my_model")
        '''
    
    @staticmethod
    def reshape_state(state):
        return np.array(state).reshape([1, -1])
