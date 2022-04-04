from random import seed
from rlsystem import RLSystem
from tournament import Tournament
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)


def main():
    # TESTS
    #rl_system_test = RLSystem()
    #rl_system_test.algorithm()

    tournament = Tournament()
    tournament.play_tournament()

main()
