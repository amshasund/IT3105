import os
import numpy as np
from parameters import (
    hex_board_size,
    hidden_layers,
    activation_function,
    optimizer,
    temperature,
    batch_size,
    epochs
)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


class ANet:
    def __init__(self):
        self.num_input_nodes = hex_board_size**2
        self.model = None

    def build_model(self, loss=tf.keras.losses.categorical_crossentropy):
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
        model.add(tf.keras.layers.Dense(
            hex_board_size**2, activation='softmax'))

        model.compile(optimizer=optimizer, loss=loss,
                      metrics=[
                          tf.keras.metrics.categorical_accuracy])
        self.model = model

    def train_model(self, rbuf, read_from_file=False):
        # Code only used when training on created data set to find
        # best parameters for our networks
        if read_from_file:
            with open(rbuf) as file_in:
                replay = []
                for line in file_in:
                    replay.append(eval(line.strip("\n")))
        else:
            replay = list(rbuf)
        
        # Reformat data and split into states to train on 
        # with previus states as targets
        arrays = list(map(list, zip(*replay)))
        states = np.array(arrays[0])
        targets = np.array(arrays[1])
        
        # Normalize targets to avoid overflow
        targets = [distribution/sum(distribution) for distribution in targets]
        # "One-hot encode" targets with a temperature
        targets = [distribution**(1/temperature)
                   for distribution in targets]
        # Normalize targets again
        targets = [distribution/sum(distribution) for distribution in targets]

        X,Y = [],[]
        for i in range(len(states)):
            x,y = states[i], targets[i]
            # Flip state and targets for vertical player
            if states[i][0] == -1:
                y = self.flip_distribution(y)
            x = self.flip_state(x)

            X.append(x)
            Y.append(y)
        
        X = np.array(X)
        Y = np.array(Y)

        # Train model and print the cross entropy loss
        history = self.model.fit(X, Y, shuffle=True, batch_size=batch_size, epochs=epochs, verbose=0)
        print(history.history['loss'])
        
    def choose_action(self, state, model, legal_actions):
        # Flip state
        state = self.flip_state(state)

        # Make state ready for input to actor net model (shape=[[]])
        state_to_model = self.reshape_state(state)

        # Get preference distribution from actor net model
        distribution = np.array(model.predict(state_to_model)[0])

        # Flip back
        if state[0] == -1:
            distribution = self.flip_distribution(distribution)

        # Eliminate illegal moves
        distribution = distribution * legal_actions

        # Normalize distribution to ensure no error from np.random.choice
        # For cases when distribution is all zeroes
        if (np.sum(distribution) != 0):
            distribution = distribution / (np.sum(distribution))
        else:
            distribution = legal_actions / (np.sum(legal_actions))

        # Find 1d index of flattened distribution
        return np.random.choice(range(len(distribution)), p=distribution)

    def save_model(self, game_nr):
        ''' Calling `save('my_model')` creates a SavedModel folder `my_model`.
        model.save("my_model")  It can be used to reconstruct the model identically.
        reconstructed_model = keras.models.load_model("my_model")'''
        self.model.save("best_models/ada_lovelace_model_3x3_{nr}.h5".format(nr=game_nr))
        
    def flip_state(self, state):
        board =  np.array(state[1:])
        board = board.reshape((hex_board_size, hex_board_size))
        # Flip board for vertical player
        if state[0] == -1:
            board = board.T*(-1)   # *(-1) to flip player of pieces to train as player 1
        return board.flatten()

    def flip_distribution(self, dist):
        dist = dist.reshape((hex_board_size, hex_board_size))
        # Flip distribution for vertical player
        dist = dist.T
        return dist.flatten()

    @staticmethod
    def reshape_state(state):
        # Reshape state into the wanted type for tensor flow [[]]
        return np.array(state).reshape([1, -1])

# To make MCTS run faster
class LiteModel:
    """Source: https://micwurm.medium.com/using-tensorflow-lite-to-speed-up-predictions-a3954886eb98"""

    @classmethod
    def from_file(cls, model_path):
        return LiteModel(tf.lite.Interpreter(model_path=model_path))

    @classmethod
    def from_keras_model(cls, kmodel):
        converter = tf.lite.TFLiteConverter.from_keras_model(kmodel)
        tflite_model = converter.convert()
        return LiteModel(tf.lite.Interpreter(model_content=tflite_model))

    def __init__(self, interpreter):
        self.interpreter = interpreter
        self.interpreter.allocate_tensors()
        input_det = self.interpreter.get_input_details()[0]
        output_det = self.interpreter.get_output_details()[0]
        self.input_index = input_det["index"]
        self.output_index = output_det["index"]
        self.input_shape = input_det["shape"]
        self.output_shape = output_det["shape"]
        self.input_dtype = input_det["dtype"]
        self.output_dtype = output_det["dtype"]

    def predict(self, inp):
        inp = inp.astype(self.input_dtype)
        count = inp.shape[0]
        out = np.zeros((count, self.output_shape[1]), dtype=self.output_dtype)
        for i in range(count):
            self.interpreter.set_tensor(self.input_index, inp[i:i+1])
            self.interpreter.invoke()
            out[i] = self.interpreter.get_tensor(self.output_index)[0]
        return out

    def predict_single(self, inp):
        """ Like predict(), but only for a single record. The input data can be a Python list. """
        inp = np.array([inp], dtype=self.input_dtype)
        self.interpreter.set_tensor(self.input_index, inp)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_index)
        return out[0]
