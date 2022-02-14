import numpy as np

from parameters import discs, pegs


class Disc:
    def __init__(self, size):
        self.size = size


class Peg:
    def __init__(self, peg_nr):
        self.peg_nr = peg_nr


class TowersPlayer:
    def __init__(self, env):
        self.env = env
        # State
        self.game_board = self.env.game_board
        self.num_moves = 0
        self.reward = 0

    def get_game_board(self):
        return self.game_board

    def get_reward(self):
        return 15 - self.num_moves

    def place_disc(self, disc_nr, peg_nr):
        pass

    def update_game_board(self):
        pass

    def set_start_game_board(self):
        pass

    def get_legal_options(self):
        pass


class TowersEnv:
    def __init__(self):
        self.game_board = self.create_game_board()
        self.peg_list = self.create_pegs()
        self.disc_list = self.create_discs()

    def reset_environment(self):
        pass

    def update_game_board(self):
        pass

    def get_options(self):
        pass

    @staticmethod
    def create_game_board():
        # Create gameboard with the shape from numbers of pegs and discs
        game_board = np.zeros((discs, pegs), dtype=int)

        # Sett all discs in first peg on game board
        for i in range(discs):
            game_board[i][0] = i + 1
        return game_board

    def print_game_board(self):
        game_board = self.game_board
        print('\n'.join(['\t'.join([(str(cell) if cell != 0 else '*') for cell in row]) for row in game_board]))

    @staticmethod
    def create_pegs():
        peg_list = []
        for i in range(1, pegs + 1):
            print("Peg created! Peg nr: " + str(i))
            peg_list.append(Peg(i))
        return peg_list

    @staticmethod
    def create_discs():
        disc_list = []
        for i in range(1, discs + 1):
            print("Disc created! Disc with size: " + str(i))
            disc_list.append(Peg(i))
        return disc_list


class TowersWorld:
    def __init__(self):
        self.environment = TowersEnv()
        self.player = TowersPlayer(self.environment)

    def get_actions(self):
        pass

    def get_state(self):
        pass

    def get_all_possible_states(self):
        pass

    def get_possible_actions_from_state(self):
        pass

    def do_action(self, action):
        pass

    def get_reward(self):
        pass

    def is_game_over(self):
        pass

    @staticmethod
    def print_results(policy):
        pass
