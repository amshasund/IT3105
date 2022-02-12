import numpy as np

from parameters import discs, pegs


class Disc:
    pass


class Peg:
    pass


class TowersPlayer:
    pass


class TowersEnv:
    def __init__(self):
        self.game_board = self.create_game_board()

        # self.peg_list = self.create_pegs()
        # self.disc_list = self.create_discs()

    def create_game_board(self):
        game_board = np.zeros((discs, pegs), dtype=int)
        print("hallo")
        # Sett all discs in first peg on game board
        for i in range(discs):
            game_board[i][0] = i + 1
        return game_board

    def print_game_board(self):
        game_board = self.game_board
        print('\n'.join(['\t'.join([(str(cell) if cell != 0 else '*') for cell in row]) for row in game_board]))

    def create_pegs(self):
        pass

    def create_discs(self):
        pass


class TowersWorld:
    def __init__(self):
        self.environment = TowersEnv()
        # self.player = TowersPlayer(self.environment)
