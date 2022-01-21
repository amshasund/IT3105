import random
from ..parameters import win_probability


class GamblerPlayer:
    def __init__(self, env):
        self.units = random.randint(1, 99)
        self.env = env

    def get_units(self):
        return self.units

    def place_bet(self, bet):
        self.units += self.env.perform_bet(bet)

    def get_possible_bets(self):
        return self.env.get_legal_bets(self.units)


class Coin:
    def __init__(self):
        self.win_probability = win_probability

    def flip(self):
        return random.choices(
            [False, True],
            weights=(1 - win_probability, win_probability),
            k=1
        )


class GamblerEnv:
    def __init__(self):
        self.coin = Coin()

    @staticmethod
    def get_range_of_units():
        return list(range(0, 100 + 1))

    def perform_bet(self, bet):
        flip_result = self.coin.flip()
        if flip_result:
            return bet
        return -bet

    def get_legal_bets(self, units):
        if units > 50:
            return list(range(1, 100 - units + 1))
        return list(range(1, units + 1))


class GamblerWorld:
    def __init__(self):
        self.environment = GamblerEnv()
        self.player = GamblerPlayer(self.environment)

    def get_actions(self):
        self.player.get_possible_bets()

    def get_state(self):
        self.player.get_units()

    def get_all_possible_states(self):
        return self.environment.get_range_of_units()

    def get_possible_actions_from_state(self, state):
        # in this game, state equals amount of units
        return self.environment.get_legal_bets(state)

    def do_action(self, action):
        self.player.place_bet(action)
