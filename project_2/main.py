import tensorflow as tf
import os
import numpy as np
from random import seed
from parameters import model_name
from rlsystem import RLSystem
from tournament import Tournament, play_model_against_random
from anet import ANet
from mct import MonteCarloTree
from hex import StateManager
import random
import absl.logging

# Does not init GPU's
os.environ['CUDA_VISIBLE_DEVICES'] = ''
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    # Train
    rl_system_test = RLSystem()
    rl_system_test.algorithm()

    # Tournament
    tournament = Tournament()
    tournament.play_tournament()

    # Random
    model = tf.keras.models.load_model(model_name)
    play_model_against_random(model)


main()
