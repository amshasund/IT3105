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
        return 0

    def move_disc(self, disc_nr, peg_nr):
        self.env.update_game_board(disc_nr, peg_nr)

    def set_start_game_board(self):
        self.env.reset_environment()

    def get_legal_options(self):
        return self.env.get_options()


class TowersEnv:
    def __init__(self):
        self.game_board = self.create_game_board()

    def reset_environment(self):
        self.game_board = self.create_game_board()

    def update_game_board(self, disc_nr, peg_nr):
        # Remove the disc from a peg
        for peg in self.game_board:
            if peg:
                if peg[-1] == disc_nr:
                    peg.pop()

        # Add the disc to a new peg
        self.game_board[peg_nr].append(disc_nr)

    def get_options(self):
        options = []
        top_discs = []

        # Find last disc or element in every peg
        for peg in self.game_board:
            if not len(peg) == 0:
                top_discs.append(peg[-1])
            else:
                top_discs.append(0)

        # Find possible pegs for all top discs
        for peg_nr, top_disc in enumerate(top_discs):
            if top_disc != 0:
                for next_peg_nr, next_top_disc in enumerate(top_discs):
                    if not peg_nr == next_peg_nr:
                        if next_top_disc == 0:
                            options.append((top_disc, next_peg_nr))
                        elif top_disc < next_top_disc:
                            options.append((top_disc, next_peg_nr))
        return options

    @staticmethod
    def create_game_board():
        # Create gameboard as a list of lists
        game_board = []

        # Add all disc to the first peg
        peg_1 = list(range(discs, 0, -1))
        game_board.append(peg_1)

        # Add the rest of the pegs as empty lists
        for i in range(pegs - 1):
            game_board.append([])
        return game_board

    def print_game_board(self):
        game_board = self.game_board
        print('\n'.join(['\t'.join([(cell * "*" if cell != 0 else '|')
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
