import os
import numpy as np
from random import seed
from rlsystem import RLSystem
from tournament import Tournament
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
    rl_system_test = RLSystem()
    rl_system_test.algorithm()
    
    tournament = Tournament()
    tournament.play_tournament()

    # Test
    #anet = ANet()
    #anet.build_model()

    #anet.train_model("rbuf.txt", True)
    #anet.save_model(666)

    """ 
    for _ in range(0, 10):
        manager = StateManager()
        #mct = MonteCarloTree(None)
        #anet = ANet()
        model = tf.keras.models.load_model("models/please_model_4x4_666.h5")
        game = manager.start_game()

        while not manager.is_final(game):
            state = manager.get_state(game)
            if manager.get_next_player(state) == 1:
                # Network models
                state_init = manager.get_state(game)
                legal_actions = manager.get_legal_actions(game)
                best_action = anet.choose_action(
                    state_init, model, legal_actions)
                manager.do_action(game, best_action)
            else:
                # Random
                action = random.choice(np.argwhere(
                    manager.get_legal_actions(game) == 1).reshape(-1))
                manager.do_action(game, action)
        winner = manager.is_final(game)
        assert winner is not False
        print(winner)
    """


main()
