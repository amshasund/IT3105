# From pseudocode
from anet import ANet
from mct import MonteCarloTree
from hex import Hex
import copy
import numpy as np

from parameters import (
    number_actual_games,
    number_search_games,
    save_interval,
)


class RLSystem:
    def __init__(self):
        self.hex = Hex()
        self.anet = ANet()
        self.mct = MonteCarloTree(self.anet)

    def algorithm(self):
        # Intialize replay buffer
        replay_buffer = []

        # Create neural nel
        self.anet.build_model()

        # Play games
        for actual_game in range(number_actual_games):
            # Initialize actual game board
            self.hex.init_game_board()

            # Set start state: [board, last_move]
            state_init = self.hex.get_state()

            # TODO: init mct to a single root, which represents s_init
            root_state = copy.deepcopy(state_init)
            self.mct.init_tree(root_state)
            root = self.mct.get_root()

            while not self.hex.game_over():
                hex_mc = Hex()
                hex_mc.set_game_state(root.get_state())

                # TODO: implement this using algorithm 1 from materials ref studass
                self.mct.search(hex_mc, root)

                visit_dist = self.mct.get_distribution(root, self.hex.get_legal_moves())
                replay_buffer.append((root, visit_dist))

                # Choose actual move
                # TODO: Send in reformatted board state
                new_position=np.unravel_index(np.argmax(np.array(visit_dist).flatten()), np.array(root.get_board()).shape)
                actual_move = [self.hex.get_next_player(), list(new_position)]

                # Perform move
                self.hex.perform_move(actual_move, print=True)
                successor_board = self.hex.get_non_object_board(self.hex.get_hex_board())
                self.mct.retain_and_discard(successor_board)
                
                root = self.mct.get_root()
            print("Game finished")
            self.anet.train(replay_buffer)
            # Save parameters for tournament
            if actual_game % save_interval == 0:
                self.anet.save_parameters()
