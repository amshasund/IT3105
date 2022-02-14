import numpy as np

from parameters import discs, pegs


class TowersPlayer:
    def __init__(self, env):
        self.env = env
        self.num_moves = 0
        self.reward = 0

    def get_game_board(self):
        return self.env.game_board

    def get_reward(self):
        # TODO: -1 per and 100 for win?
        return 15 - self.num_moves

    def move_disc(self, disc_nr, peg_nr):
        self.env.update_game_board(disc_nr, peg_nr)

    def set_start_game_board(self):
        self.env.create_game_board()

    def get_legal_options(self):
        return self.env.get_options()


class TowersEnv:
    def __init__(self):
        self.game_board = self.create_game_board()
        self.print_game_board()

    def reset_environment(self):
        self.create_game_board()

    def update_game_board(self, disc_nr, peg_nr):
        result = np.where(self.game_board == disc_nr)
        self.game_board

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
        print('\n'.join(['\t'.join([(cell*"*" if cell != 0 else '|')
              for cell in row]) for row in game_board]))


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
    def print_end_results(policy):
        pass

    def print_episode(self):
        pass

    def save_episode_for_print(self):
        pass
