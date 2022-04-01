import numpy as np
import tensorflow as tf
import random

from parameters import (
    hex_board_size,
    hidden_layers,
    activation_function,
    optimizer, 
    starting_player,
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

    def train_model(self, case):
        # Input: [Node, visit_dist]
        # Output: probability distribution over all legal moves
        node = case[0]
        board = node.get_non_object_board()
        print(board)
        player = node.get_player()
        print(player)
        state = np.insert(board, 0, player)
        print(state)
        state = self.reshape_state(state)
        print(state)
        
        # Normalize replay buffer data
        target = case[1].flatten() / case[1].flatten().sum()
        target = self.reshape_state(target)

        print(target)
        

        
        self.model.fit(np.array(state, dtype=float), np.array(target, dtype=float), verbose=0)


    def rollout(self, state, legal_moves):
        player = state[1]
        board = state[0]
        
        # -1 og 1 bedre enn 1 og 2
        # input: [1 0 0 0 0 0 0 0 0 0] means that player 1 starts with clean board
        state = np.insert(board, 0, player)

        # Make state ready for input to actor net model
        state = self.reshape_state(state)

        # Get preference distribution from actor net model
        distribution = np.array(self.model(state)[0])

        # Eliminate illegal moves
        dist_move = distribution * np.array(legal_moves).flatten()

        # Normalize dist_move to ensure no error from np.random.choice
        dist_move = dist_move / np.sum(dist_move)
        
        # When choosing move, use prob from anet to choose a move [0.4, 0.45, 0.1, 0.05]
        # Find 1d index of flattened dist_move
        chosen_move_flattened = np.random.choice(range(len(dist_move)), p=dist_move)
        
        # Find 2d index from 1d index based on shape of legal_moves
        chosen_move = np.unravel_index(chosen_move_flattened, np.array(legal_moves).shape)
        
        return [player, chosen_move]
    
    @staticmethod
    def reshape_state(state):
        return np.array(state).reshape([1, -1])
