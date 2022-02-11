from sw_the_gambler import GamblerWorld
from sw_pole_balancing import PoleWorld
from sw_towers_of_hanoi import TowersWorld
from parameters import game


class SimWorld(
    GamblerWorld if game == 'the_gambler' else (
        PoleWorld if game == 'pole_balancing' else (
            TowersWorld if game == 'towers_of_hanoi' else
            False))):
    pass
