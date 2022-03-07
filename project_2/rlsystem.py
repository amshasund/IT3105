# From pseudocode
from anet import ANet
from mcts import MCTS
from hex import Hex

from parameters import (
    number_actual_games,
    number_search_games,
    save_interval,
)


class RLSystem:
    def __init__(self):
        self.hex = Hex()
        self.mcts = MCTS()
        self.anet = ANet()

    def algorithm(self):
        rbuf = [] 
        self.anet.build_model()
        
        # Play games
        for actual_game in range(number_actual_games):
            self.hex.init_game_board()
            state_init = self.hex.get_state()  # TODO: set s_init to starting board state
            # TODO: init mct to a single root, which represents s_init
            self.mcts.init_tree(state_init)
            root = state_init
            
            while not self.hex.game_over():
                hex_mc = Hex(root)
                
                self.mcts.search_tree() #TODO: implement thios using algorithm 1 from materials ref studass
                
                visit_dist =  self.mcts.get_distribution(root)
                rbuf.append((root, visit_dist))

                # Choose actual move
                actual_move = self.anet.choose_move(visit_dist)

                # Perform move
                self.hex.perform_move(actual_move)
                successor_state = self.hex.get_state()
                self.mcts.retain_and_discard(successor_state)
                root = successor_state
            
            self.anet.train(rbuf)
            # Save parameters for tournament
            if actual_game % save_interval == 0:
                self.anet.save_parameters()
