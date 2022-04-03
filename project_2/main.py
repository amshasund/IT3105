from random import seed
from rlsystem import RLSystem
from tournament import Tournament


def main():
    # TESTS
    #rl_system_test = RLSystem()
    #rl_system_test.algorithm()

    tournament = Tournament()
    tournament.play_tournament()

main()
