from games.the_gambler import TheGamblerEnvironment
from games.pole_balancing import PoleBalancingEnvironment
from games.towers_of_hanoi import TowersEnvironment
from parameters import game


class SimWorld(TheGamblerEnvironment if game == 'the_gambler' else (
        PoleBalancingEnvironment if game == 'pole_balancing' else (
                TowersEnvironment if game == 'towers_of_hanoi' else False))):

    def __init__(self):
        super().__init__()

    def get_actions(self):
        super().get_actions(self)

    def get_state(self):
        pass
