import numpy as np
import tensorflow as tf

from parameters import (
    hex_board_size,
    hidden_layers,
    activation_function,
    optimizer,
)


class ANet:
    def __init__(self):
        self.num_input_nodes = hex_board_size**2 + \
            1  # one extra input to know which player
        self.model = None

    # loss: cross-entropy, output for each move

    # This comparison of output and target probablity distributions
    # normally calls for a cross-entropy loss/error function; and to
    # produce a probability distribution in the output vector, the
    # softmax activation function is recommended.

    def build_model(self, loss=tf.keras.losses.MeanSquaredError()):
        # Create Neural Net
        model = tf.keras.models.Sequential()

        # TODO: No need for this
        # Initialize weights and biases
        initializer = tf.keras.initializer.TruncatedNormal(
            mean=0., stddev=1.)  # TODO

        # Add input layer
        model.add(tf.keras.layers.Input(shape=(self.num_input_nodes,)))  # TODO

        # Add hidden layers
        for i in range(len(hidden_layers)):
            model.add(tf.keras.layers.Dense(
                hidden_layers[i],
                activation=activation_function[i],
                kernel_initializer=initializer,       # TODO
                use_bias=True,                        # TODO
                bias_initializer=initializer))        # TODO

        # Add output layer
        # TODO: output size same as amount of possible moves (hex board size)
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))  # TODO

        # Using stochastic gradient descent when compiling
        # Optimizer is a string, could be a problem that it is not a tf function
        model.compile(optimizer=optimizer, loss=loss,
                      metrics=[
                          tf.keras.metrics.categorical_accuracy])

        self.model = model
