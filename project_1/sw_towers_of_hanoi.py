import copy
import time

import matplotlib.pyplot as plt

from parameters import discs, pegs, max_steps, episodes, frame_delay


class TowersPlayer:
    def __init__(self, env):
        self.env = env
        self.num_moves = 0
        self.reward = 0

    def get_game_board(self):
        return self.env.game_board

    def get_reward(self):
        if self.is_game_over() and not self.num_moves >= max_steps:
            self.reward = 20

        else:
            self.reward = -1

        return self.reward

    def move_disc(self, action):
        str_action = str(action)
        disc_nr = int(str_action[0])
        peg_nr = int(str_action[1])
        self.env.update_game_board(disc_nr, peg_nr)
        self.num_moves += 1

    def reset_player(self):
        self.num_moves = 0

    def get_legal_options(self, state=None):
        return self.env.get_options(state)

    def is_game_over(self):
        if self.num_moves >= max_steps:

            return True
        else:
            state = self.get_game_board()
            for peg in state[1:]:
                if len(peg) == discs:
                    return True
        return False


class TowersEnv:
    def __init__(self):
        self.game_board = self.create_game_board()

    def reset_environment(self):
        self.game_board = self.create_game_board()

    def update_game_board(self, disc_nr, peg_nr):
        self.game_board = copy.deepcopy(self.game_board)
        # Remove the disc from a peg
        for peg in self.game_board:
            if peg:
                if peg[-1] == disc_nr:
                    peg.pop()

        # Add the disc to a new peg
        self.game_board[peg_nr].append(disc_nr)

    def get_options(self, state=None):
        options = []
        top_discs = []
        if state:
            game_board = state
        else:
            game_board = self.game_board

        # Find last disc or element in every peg
        for peg in game_board:
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
                            # save actions as integer "top_disc" + "next_peg_nr"
                            # ex: move disc 5 to peg 3: options.append(53)
                            options.append(
                                int(str(top_disc) + str(next_peg_nr)))
                        elif top_disc < next_top_disc:
                            options.append(
                                int(str(top_disc) + str(next_peg_nr)))
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

    def print_game_board(self, state=None):
        if state:
            game_board = list(list(s) for s in state)
        else:
            game_board = copy.deepcopy(self.game_board)

        for peg in game_board:
            while len(peg) < discs:
                peg.append(0)
            peg.reverse()
        string = ""
        for i in range(len(game_board[0])):
            for peg in game_board:
                if peg[i] == 0:
                    string += '{:^10s}'.format(" ")
                else:
                    string += '{:^10s}'.format(peg[i]
                                               * "*")
            string += "\n"
        string += '{:^10s}'.format("'") * len(game_board)
        print(string)


class TowersWorld:
    def __init__(self):
        self.environment = TowersEnv()
        self.player = TowersPlayer(self.environment)
        self.moves_per_episode = [0] * (episodes + 1)
        self.states_for_current_episode = []

    def get_actions(self):
        return self.player.get_legal_options()

    def get_state(self, state_type=None):
        state = copy.deepcopy(self.player.get_game_board())
        return self.state_to_send(state)

    def get_possible_actions_from_state(self, state_received):
        state = self.received_to_state(state_received)
        return self.player.get_legal_options(state)

    @staticmethod
    def state_to_send(state):
        send = [-1] * discs
        for peg_num in range(len(state)):
            for disc in state[peg_num]:
                send[disc - 1] = peg_num

        send = tuple(send)
        return send

    @staticmethod
    def received_to_state(received):
        state = []
        state_received = list(received)
        # Create the state shape
        for p in range(pegs):
            state.append(copy.deepcopy([]))

        for index, peg in enumerate(state_received[::-1]):
            # Reversed list - must reverse indices
            disc = discs - index
            state[peg].append(disc)
        return state

    def do_action(self, action):
        self.player.move_disc(action)

    def get_reward(self):
        return self.player.get_reward()

    def reset_sim_world(self):
        self.environment.reset_environment()
        self.player.reset_player()

    def is_time_out(self):
        return self.player.num_moves >= max_steps

    def is_game_over(self):
        is_over = self.player.is_game_over()
        if is_over:
            print("You used ", self.player.num_moves, " moves")
        return is_over

    def save_history(self, episode, str_states):
        self.moves_per_episode[episode] = self.player.num_moves
        self.states_for_current_episode = str_states

    def print_end_results(self, policy):
        # Plot: The Progression of Learning
        x = list(range(1, episodes + 1))
        y = self.moves_per_episode[1:]

        plt.plot(x, y)
        plt.xlabel("Episode")
        plt.ylabel("Number of Moves")
        plt.title("The Progression of Learning")
        plt.show()

    def print_episode(self, episode):
        max = 20
        counter = 0
        for state_tuple in self.states_for_current_episode:
            state = self.received_to_state(state_tuple)
            if counter < max:
                self.environment.print_game_board(state)
                counter += 1
            else:
                state = self.received_to_state(self.states_for_current_episode[-1])
                print("Final state:")
                self.environment.print_game_board(state)
                print("Num moves ", len(self.states_for_current_episode))
                return
            time.sleep(frame_delay)
