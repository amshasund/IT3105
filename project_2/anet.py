import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
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

        # TODO: use minibatch from keras
        # make array of states and targets
        # shuffle

        # rbuf = [(x1,y1),(x2,y2), ...]
        # x_array = 
        # y_array = 
        # Train on one-hot of argmax of target
        # [5, 3, 2]
        # []**(1/T) -> T = 0.1 (T mot null, tilsvarer nesten one-hot)
        # opphøy i 10 og normaliser på nytt 
        # beware of integer overflow <- normaliser array før opphøying
        for key in rbuf:
            # Train on random batches of rbuf
            if np.random.choice([True, False], p=[0.5, 0.5]):
                state = np.array(key, dtype=float)
                state = self.reshape_state(state)
                target = np.array(rbuf[key], dtype=float).flatten()
                
                # Normalize replay buffer data
                target = target / target.sum()
                target = self.reshape_state(target)

                # kan også flippe 180 grader for å trene meir
                
                self.model.fit(state, target, verbose=0) # default batchsize 32 elements, use shuffle


    def choose_action(self, state, model, legal_actions):
        # Make state ready for input to actor net model
        state_to_model = self.reshape_state(state)
        # [1 0 0 0 0 0 0 0 0 0]
        # [-1 0 1 0 0 0 0 0 0 0]
        # [[010],[001],[100],[010]]
        # hex triks: board.T*(-1)
        # TODO: make utility function
        # husk å flippe tilbake etter valg av action

        # Get preference distribution from actor net model
        distribution = np.array(model.predict(state_to_model)[0])

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
        self.model.save("models/cool_model3x3_{nr}.h5".format(nr=game_nr))

        '''
        # Calling `save('my_model')` creates a SavedModel folder `my_model`.
        model.save("my_model")

        # It can be used to reconstruct the model identically.
        reconstructed_model = keras.models.load_model("my_model")
        '''
    
    @staticmethod
    def reshape_state(state):
        return np.array(state).reshape([1, -1])


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
