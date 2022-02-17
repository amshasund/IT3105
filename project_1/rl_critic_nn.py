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
        model.add(tf.keras.layers.Input(shape=(self.num_input_nodes,)))

        # Add hidden layers
        for i in range(len(neural_dim)):
            model.add(tf.keras.layers.Dense(neural_dim[i], activation=act))

        # Add output layer
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        # Using stochastic gradient descent when compiling
        model.compile(optimizer=opt(learning_rate=lr_critic), loss=loss,
                      metrics=[
                          tf.keras.metrics.categorical_accuracy])

        self.nn_model = model

    def train_model(self, reward, state, new_state):
        state = self.reshape_state(state)
        target = reward + discount_factor_critic * self.get_value(new_state)
        self.nn_model.fit(state, target, verbose=0)

    def get_value(self, state):
        state = self.reshape_state(state)
        value = self.nn_model(state)
        return value

    def get_TD_error(self):
        return self.TD_error

    def set_TD_error(self, reward, state, new_state, game_over, time_out):
        self.TD_error = reward + discount_factor_critic * \
                        self.get_value(new_state) * \
                        (game_over == time_out) - self.get_value(state)

    @staticmethod
    def reshape_state(state):
        return np.array(state).reshape([1, -1])
