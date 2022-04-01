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
    print_games
)


class RLSystem:
    def __init__(self):
        self.hex = Hex()
        self.anet = ANet()
        self.mct = MonteCarloTree(self.anet)

    def algorithm(self):
        # Intialize replay buffer
        replay_buffer = dict()

        # Create neural nel
        self.anet.build_model()

        # Play games
        for actual_game in range(1, number_actual_games+1):
            # Initialize actual game board
            self.hex.init_game_board()

            last_game = (True if actual_game in print_games else False)
            if last_game:
                print("START GAME \n")
                self.hex.print_game_board()

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
                # TODO: Make dictionary and have non_object board state as key
                replay_buffer = self.add_to_rbuf(replay_buffer, root, visit_dist)

                # Choose actual move
                # TODO: Send in reformatted board state
                new_position=np.unravel_index(np.argmax(np.array(visit_dist).flatten()), np.array(root.get_board()).shape)
                actual_move = [self.hex.get_next_player(), list(new_position)]

                # Perform move
                self.hex.perform_move(actual_move, print=last_game)
                successor_board = self.hex.get_non_object_board(self.hex.get_hex_board())
                self.mct.retain_and_discard(successor_board)
                
                root = self.mct.get_root()
            if last_game:
                print("WINNER: Player", self.hex.game_over())
            self.anet.train_model(replay_buffer)
            
            # Save parameters for tournament
            if actual_game % save_interval == 0:
                self.anet.save_model(actual_game)

    def add_to_rbuf(self, rbuf, root, dist):
        board = np.array(root.get_non_object_board()).flatten()
        board = np.insert(board, 0, root.get_player())
        key = tuple(board)
        if rbuf and key in rbuf:
            rbuf[key] += dist
        else:
            rbuf[key] = dist
        return rbuf
