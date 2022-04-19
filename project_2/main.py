import os

import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from parameters import model_name, number_actual_games
from rlsystem import RLSystem
from tournament import Tournament, play_model_against_random

# Does not init GPU's
os.environ['CUDA_VISIBLE_DEVICES'] = ''


def main():
    # Train
    rl_system_test = RLSystem()
    rl_system_test.algorithm()

    # Tournament
    tournament = Tournament()
    tournament.play_tournament()

    # Random
    model = tf.keras.models.load_model(model_name.format(number_actual_games))
    play_model_against_random(model)


main()
