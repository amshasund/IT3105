import numpy as np
import tensorflow as tf

from parameters import (
    lr_critic,
    discount_factor_critic,
    neural_dim
)


class CriticNN:
    def __init__(self):
        self.num_input_nodes = 0
        self.nn_model = None
        self.TD_error = 0

    def set_num_input_nodes(self, num_input_nodes):
        self.num_input_nodes = num_input_nodes

    def build_model(self, act='relu', opt=tf.keras.optimizers.SGD, loss=tf.keras.losses.MeanSquaredError()):
        # Create Neural Net
        model = tf.keras.models.Sequential()

        # Add input layer
        model.add(tf.keras.layers.Input((self.num_input_nodes),))

        # Add hidden layers
        for i in range(len(neural_dim)):
            model.add(tf.keras.layers.Dense(neural_dim[i], activation=act))

        # Add output layer
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        # Using stochastic gradient descent when compiling
        model.compile(optimizer=opt(learning_rate=lr_critic), loss=loss,
                      metrics=[
                          tf.keras.metrics.categorical_accuracy])  # Keith had this

        self.nn_model = model

    def train_model(self, reward, state, new_state):
        # TODO: train with list of states and targets
        state_bin = self.state_to_binary(state)
        target = reward + discount_factor_critic * self.get_value(new_state)
        self.nn_model.fit(state_bin, target, verbose=0)

    def get_value(self, state):
        state_bin = self.state_to_binary(state)
        value = float(sum(self.nn_model(state_bin)).numpy())
        # print(value)
        # print(tf.keras.backend.eval(value))
        return self.nn_model(state_bin)

    def get_TD_error(self):
        return self.TD_error

    def set_TD_error(self, reward, state, new_state, game_over):
        self.TD_error = reward + discount_factor_critic * \
            float(sum(self.get_value(new_state)).numpy()) * \
            (not game_over) - float(sum(self.get_value(state)).numpy())
        # print("Value: ", self.get_value(state),"TD: ", self.get_TD_error())

    @staticmethod
    def state_to_binary(state):
        """ Returns a binary representation of the state
            ex: (4,2,1) -> '421' -> 421 -> 110100101 """
        binary_state = []

        # For state being tuple -> Towers of Hanoi or Pole Balancing
        if isinstance(state, tuple):
            for element in state:

                # For tuple of tuples -> Towers of Hanoi
                if isinstance(element, tuple):
                    if len(element) == 0:
                        binary_state.append(0)
                    else:
                        values_string = ''.join(map(str, element))
                        binary_state.append(
                            int(np.binary_repr(int(values_string), 8)))

                # For tuples of lists -> Pole Balancing
                else:
                    binary_state.append(int(np.binary_repr(int(element), 8)))

        # For state being an int -> The Gambler
        else:
            binary_state.append(int(np.binary_repr(int(state), 8)))
        # .reshape(1, -1) does not work for towers of hanoi
        # print(np.array(binary_state))
        return np.array(binary_state)  # .reshape(1,-1)
