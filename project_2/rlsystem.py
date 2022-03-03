# From pseudocode
from anet import ANet
from mcts import MCTS
from hex import Hex

from parameters import (
    number_actual_games,
    number_search_games
)


class RLSystem:
    def __init__(self):
        self.hex = Hex()
        self.mcts = MCTS()
        self.anet = ANet()

    def algorithm(self):
        interval_s = 0  # TODO: save interval for ANET parameters
        rbuf = 0  # TODO: clear replay buffer (rbuf)
        self.anet.init_parameters()  # TODO: randomly initialize weigths and biases of ANET
        # TODO: What is this number?
        for actual_game in range(number_actual_games):
            self.hex.init_game_board()  # TODO: initialize actual game board to an empty board
            state_init = self.hex.get_state()  # TODO: set s_init to starting board state
            # TODO: init mct to a single root, which represents s_init
            self.mcts.init_tree(state_init)
            root = state_init
            while not self.hex.game_over():
                hex_mc = Hex(root)
                for search_game in range(number_search_games):
                    is_leaf = False
                    while not is_leaf:
                        leaf, is_leaf = self.mcts.search_to_leaf(root)
                        hex_mc.update_leaf(leaf)
                    is_final = False
                    while not is_final:
                        final, is_final = self.anet.choose_rollout(leaf)
                        hex_mc.update_final(final)
                    self.mcts.perform_backpropagation(final, root)
