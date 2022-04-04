# From pseudocode
from anet import ANet, LiteModel
from mct import MonteCarloTree
from hex import StateManager
import numpy as np

from parameters import (
    number_actual_games,
    save_interval,
    print_games
)

class RLSystem:
    def __init__(self):
        self.manager = StateManager()
        self.anet = ANet()
        self.lite_model = LiteModel
        self.mct = MonteCarloTree(self.anet)

    def algorithm(self):
        # Intialize replay buffer
        replay_buffer = dict()

        # Create neural nel
        self.anet.build_model()
        
        # Save untrained net
        self.anet.save_model(0)

        # Play games
        for actual_game in range(1, number_actual_games+1):
            print("Running game: ", actual_game)
            # Initialize actual game board
            game = self.manager.start_game()

            print_game = (True if actual_game in print_games else False)
            if print_game:
                print("START GAME \n")
                self.manager.print_state(game)

            # Set start state: [player, board.flatten()]
            state_init = self.manager.get_state(game)
            self.mct.init_tree(state_init) 
            # TODO: send in litemodel of anet to use for rollout
            # __call__ if we want model(state) and not model.predict(state)
            
            while not self.manager.is_final(game):
                # Using Lite ANet model
                lite_model = self.lite_model.from_keras_model(self.anet.model)
                self.mct.search(self.manager, lite_model)
                visit_dist = self.mct.get_distribution(self.manager.get_legal_actions(game))
                replay_buffer = self.add_to_rbuf(replay_buffer, self.manager.get_state(game), visit_dist)

                # Choose actual move
                action = np.argmax(np.array(visit_dist))

                # Perform move
                self.manager.do_action(game, action, print=print_game)
                successor_state = self.manager.get_state(game)
                self.mct.retain_and_discard(successor_state)
               
            if print_game:
                print("WINNER: Player", self.manager.is_final(game))
            
            self.anet.train_model(replay_buffer) # cannot train a lightmodel, so this must be the real model
            
            # Save parameters for tournament
            if actual_game % save_interval == 0:
                self.anet.save_model(actual_game)

    def add_to_rbuf(self, rbuf, state, dist):
        # TODO: Debug this. Check that addition is great
        key = tuple(state)
        if rbuf and key in rbuf:
            rbuf[key] += np.array(dist)
        else:
            rbuf[key] = np.array(dist)
        return rbuf
