import absl.logging
import random
from hex import StateManager
from mct import MonteCarloTree
from tournament import Tournament
from rlsystem import RLSystem
from random import seed
import numpy as np
import os
# Does not init GPU's
os.environ['CUDA_VISIBLE_DEVICES'] = ''
absl.logging.set_verbosity(absl.logging.ERROR)


def main():

    # TESTS
    # rl_system_test = RLSystem()
    # rl_system_test.algorithm()

    # tournament = Tournament()
    # tournament.play_tournament()

    for _ in range(0, 10):
        manager = StateManager()
        mct = MonteCarloTree(None)

        game = manager.start_game()

        while not manager.is_final(game):
            state = manager.get_state(game)
            if manager.get_next_player(state) == 1:
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


main()
