# From pseudocode
from anet import ANet
from mct import MonteCarloTree
from hex import Hex
import copy

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
            root = copy.deepcopy(state_init)
            self.mct.init_tree(root)

            while not self.hex.game_over():
                hex_mc = Hex()
                hex_mc.set_game_state(root)

                # TODO: implement this using algorithm 1 from materials ref studass
                self.mct.search(hex_mc)

                visit_dist = self.mct.get_distribution(root)
                replay_buffer.append((root, visit_dist))

                # Choose actual move
                # TODO: Send in reformatted board state
                actual_move = self.anet.choose_move(visit_dist)

                # Perform move
                self.hex.perform_move(actual_move)
                successor_state = self.hex.get_state()
                self.mct.retain_and_discard(successor_state)
                root = copy.deepcopy(successor_state)

            self.anet.train(replay_buffer)
            # Save parameters for tournament
            if actual_game % save_interval == 0:
                self.anet.save_parameters()
