import numpy as np
import tensorflow as tf
import h5py

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

    def train_model(self, rbuf):
        # Input: dict{  key=tuple(player, board.flatten()), 
        #               value=board's visit distribution)}
        
        for key in rbuf:
            # Train on random batches of rbuf
            if np.random.choice([True, False], p=[0.3, 0.7]):
                state = np.array(key, dtype=float)
                state = self.reshape_state(state)
                

                target = np.array(rbuf[key], dtype=float).flatten()
                # Normalize replay buffer data
                target = target / target.sum()
                target = self.reshape_state(target)
                
                self.model.fit(state, target, verbose=0)


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
        
        # For cases when dist_move is all zeroes 
        if (np.sum(dist_move) != 0):
            dist_move = dist_move / (np.sum(dist_move))
        else:
            dist_move = np.array(legal_moves).flatten() / (np.sum(np.array(legal_moves).flatten()))
        
        # When choosing move, use prob from anet to choose a move [0.4, 0.45, 0.1, 0.05]
        # Find 1d index of flattened dist_move
        chosen_move_flattened = np.random.choice(range(len(dist_move)), p=dist_move)
        
        # Find 2d index from 1d index based on shape of legal_moves
        chosen_move = np.unravel_index(chosen_move_flattened, np.array(legal_moves).shape)
        
        return [player, chosen_move]

    def save_model(self, game_nr):
        self.model.save("super_model_{nr}.h5".format(nr=game_nr))
    
    @staticmethod
    def reshape_state(state):
        return np.array(state).reshape([1, -1])
