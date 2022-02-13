import numpy as np
import tensorflow as tf

from parameters import (
    lr_critic,
    discount_factor_critic,
    neural_dim,
    game
)


class CriticNN:
    def __init__(self, sim_world):
        self.sim_world = sim_world
        self.num_input_nodes = self.get_num_input_nodes()
        self.nn_model = self.build_model()
        self.TD_error = 0

    # We should probably do this another way
    @staticmethod
    def get_num_input_nodes():
        if game == "the_gambler":
            return 1
        elif game == "towers_of_hanoi":
            return 4
        elif game == "pole_balancing":
            return 6
        else:
            return "game is not defined correctly"

    def build_model(self, act='relu', opt=tf.keras.optimizers.SGD, loss=tf.keras.losses.MeanSquaredError()):
        # Create Neural Net
        model = tf.keras.models.Sequential()

        # Add input layer
        model.add(tf.keras.layers.Input((self.num_input_nodes,)))

        # Add hidden layers
        for i in range(len(neural_dim)):
            model.add(tf.keras.layers.Dense(neural_dim[i], activation=act))

        # Add output layer
        model.add(tf.keras.layers.Dense(1, activation=act))

        # Using stochastic gradient descent when compiling
        model.compile(optimizer=opt(learning_rate=lr_critic), loss=loss,
                      metrics=[
                          tf.keras.metrics.categorical_accuracy])  # Keith had this

        return model

    def train_model(self, reward, state, new_state):
        state_bin = self.state_to_binary(state)
        target = reward + discount_factor_critic * self.get_value(new_state)
        # print("Target: ", target)
        self.nn_model.fit(state_bin, target, verbose=0)

    def get_value(self, state):
        state_bin = self.state_to_binary(state)
        return self.nn_model(state_bin)

    def get_state(self):
        return self.sim_world.get_state()

    def get_TD_error(self):
        return self.TD_error

    def set_TD_error(self, reward, state, new_state):
        self.TD_error = reward + discount_factor_critic * \
                        self.get_value(new_state) - self.get_value(state)
        # print("Value: ", self.get_value(state),"TD: ", self.get_TD_error())

    @staticmethod
    def state_to_binary(state):
        binary_state = []
        if isinstance(state, list):
            for element in state:
                binary_state.append(int(np.binary_repr(int(element), 8)))
        else:
            binary_state.append(int(np.binary_repr(int(state), 8)))
        return np.array(binary_state)

    def binary_to_state(self, binary_state):
        state = []
        for s in binary_state:
            state.append(self.binary_to_decimal(s, 8))
        return state

    @staticmethod
    def binary_to_decimal(num, bits):
        # to handle negative numbers
        if num[0] == "1":
            return -(2 ** bits - int(num, 2))
        return int(num, 2)
