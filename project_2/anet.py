import numpy as np
import tensorflow as tf
import random

from parameters import (
    hex_board_size,
    hidden_layers,
    activation_function,
    optimizer, 
    starting_player
)


class ANet:
    def __init__(self):
        self.num_input_nodes = hex_board_size**2 + \
            1  # one extra input to know which player
        self.model = None

    def build_model(self, loss=tf.keras.losses.BinaryCrossentropy):
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
        # should not be able to move pieces to a place where another already exists.
        model.add(tf.keras.layers.Dense(
            hex_board_size**2, activation='softmax'))

        # Using stochastic gradient descent when compiling
        # Optimizer is a string, could be a problem that it is not a tf function
        model.compile(optimizer=optimizer, loss=loss,
                      metrics=[
                          tf.keras.metrics.categorical_accuracy])

        self.model = model

    def train_model(self, reward, state, new_state):
        # Input: game state + legal_moves
        # Output: probability distribution over all legal moves
        state = self.reshape_state(state)
        target = reward + discount_factor_critic * self.get_value(new_state)
        self.model.fit(state, target, verbose=0)

    def choose_move(self, state, legal_moves):
        # Explorative for rollout (behaviour (default) policy)
        # More exploitative for actual moves (target policy)
        #state = self.reshape_state(state)
        
        #value = self.model(state)
        
        if state[1]:
            player = (2 if state[1].get_player() == 1 else 1)
        else:
            player = starting_player
        # For test
        return [player, random.choice(legal_moves)]

    def reshape_state(self, state):
        """Reshapes the state from [[GameBoard], [Player]] to [gamebord1D, player]"""
        reshaped_state = []
        reshaped_state.append(state[0].flatten())
        reshaped_state.append(state[1])
        print("reshaped state", reshaped_state)
        return reshaped_state
