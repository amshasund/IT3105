import os
import numpy as np
from random import seed
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

import tensorflow as tf

def main():
    # Train
    rl_system_test = RLSystem()
    rl_system_test.algorithm()
    
    # Tournament
    tournament = Tournament()
    tournament.play_tournament()
    
    # Random
    model = tf.keras.models.load_model("best_models/basic_bitch_model_3x3_300.h5")
    play_model_against_random(model)


main()
