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
    
    #tournament = Tournament()
    #
    #tournament.play_tournament()
    # Test
    #anet = ANet()
    #anet.build_model()
    #
    """ 
    anet1 = ANet()
    anet2 = ANet()
    #model = tf.keras.models.load_model("models/luna_lovegood_model_5x5_1500.h5")
    anet1.build_model()
    anet2.build_model()
    for _ in range(0, 10):
        manager = StateManager()
        #mct = MonteCarloTree(None)
        
        anet1.train_model("luna_lovegood.txt", True)
        winners = []
        for i in range(100):
            #anet1.train_model("ada_lovelace.txt", True)
            #anet.save_model(666)
            game = manager.start_game()
            
            while not manager.is_final(game):
                state = manager.get_state(game)
                if manager.get_next_player(state) == 1:
                    # Network models
                    state_init = manager.get_state(game)
                    legal_actions = manager.get_legal_actions(game)
                    best_action = anet1.choose_action(
                        state_init, anet1.model, legal_actions)
                    manager.do_action(game, best_action)
                else:
                    # Network 2
                    state_init = manager.get_state(game)
                    legal_actions = manager.get_legal_actions(game)
                    best_action = anet2.choose_action(
                        state_init, anet2.model, legal_actions)
                    manager.do_action(game, best_action)
                    
                    # Random
                     
                    action = random.choice(np.argwhere(
                        manager.get_legal_actions(game) == 1).reshape(-1))
                    manager.do_action(game, action)
                    
            winner = manager.is_final(game)
            assert winner is not False
            winners.append(winner)
        #print(winners)
        print(sum(v==1 for v in winners))
        """

main()
