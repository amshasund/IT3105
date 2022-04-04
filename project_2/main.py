import os
import numpy as np
from random import seed
from rlsystem import RLSystem
from tournament import Tournament
from mct import MonteCarloTree
from hex import StateManager
import random
import absl.logging

# Does not init GPU's
os.environ['CUDA_VISIBLE_DEVICES'] = ''
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    # TESTS
    rl_system_test = RLSystem()
    rl_system_test.algorithm()

    tournament = Tournament()
    tournament.play_tournament()

    """
    for _ in range(0, 10):
        manager = StateManager()
        mct = MonteCarloTree(None)

        game = manager.start_game()

        while not manager.is_final(game):
            state = manager.get_state(game)
            if manager.get_next_player(state) == -1:
                state_init = manager.get_state(game)
                mct.init_tree(state_init)
                mct.search(manager, None)
                visit_dist = mct.get_distribution(
                    manager.get_legal_actions(game))
                action = np.argmax(np.array(visit_dist))
                manager.do_action(game, action)
            else:
                action = random.choice(np.argwhere(
                    manager.get_legal_actions(game) == 1).reshape(-1))
                manager.do_action(game, action)
        winner = manager.is_final(game)
        assert winner is not False
        print(winner)
    """


main()
