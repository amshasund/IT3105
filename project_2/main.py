from random import seed
from anet import ANet
from hex import Hex
from tournament import Tournament
from rlsystem import RLSystem
seed(1)

def main():
    # TESTS
    rl_system_test = RLSystem()
    rl_system_test.algorithm()
    tournament = Tournament()
    tournament.create_agents()

main()
