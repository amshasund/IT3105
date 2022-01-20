from games.the_gambler import GamblerWorld
from games.pole_balancing import PoleWorld
from games.towers_of_hanoi import TowersWorld
from parameters import game


class SimWorld(
    GamblerWorld if game == 'the_gambler' else (
        PoleWorld if game == 'pole_balancing' else (
            TowersWorld if game == 'towers_of_hanoi' else
            False))):
    pass
