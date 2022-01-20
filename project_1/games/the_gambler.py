from random import randint, choices
from ..parameters import win_probability


class TheGamblerPlayer:
    def __init__(self):
        self.units = randint(1, 99)

    def place_bet(self, bet):
        self.units += choices([-bet, bet], weights=(1 - win_probability, win_probability), k=1)


class TheGamblerEnvironment:

    def __init__(self):
        self.units = randint(1, 99)

    def get_units(self):
        return self.units

    def get_possible_bets(self):
        if self.units > 50:
            return list(range(1, 100 - self.units + 1))
        return list(range(1, self.units + 1))


class TheGamblerWorld:
    def __init__(self):
        self.environtment = TheGamblerEnvironment
        self.player = TheGamblerPlayer

    def get_actions(self):
        pass

    def