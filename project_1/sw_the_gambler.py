import random

import matplotlib.pyplot as plt

from parameters import win_probability


class Coin:
    def __init__(self):
        """ Initializes a weighted coin with a winning probability"""
        self.win_probability = win_probability

    @staticmethod
    def flip():
        """ Method that flips the weighted coin and returns the side flipped to"""
        return random.choices(
            population=[False, True],
            weights=[1 - win_probability, win_probability],
            k=1,
        )[0]


class GamblerPlayer:
    def __init__(self, env):
        """ Initializes a game player for the gambler. The player
         has an Gambler environment to play in and knows the current
         state of the game (number of units). The player also saves
         the current """
        self.env = env
        self.units = random.randint(1, 99)
        self.reward = 0

    def get_units(self):
        """ Returns the number of units (state) the player has """
        return self.units

    def get_reward(self):
        """ Returns a reword based on the state the player
        got itself into """
        # For winning
        if self.units == 100:
            self.reward = 1

        # For loosing
        elif self.units == 0:
            self.reward = -1

        # For staying alive
        else:
            self.reward = 0

        return self.reward

    def set_start_units(self):
        """ Sets a random start state of units
        between 1 and 99for the player."""
        self.units = random.randint(1, 99)

    def place_bet(self, bet):
        """ Method that places a bet by calling
        the environment with a bet request. The
        results from the betting is added to
        the players state """
        self.units += self.env.perform_bet(bet)

    def get_possible_bets(self):
        """ Method that returns legal and possible
        bets for a given state"""
        return self.env.get_legal_bets(self.units)


class GamblerEnv:
    def __init__(self):
        """ Initializes a environment for the game The Gambler.
        The environment has a object in it, which is a coin """
        self.coin = Coin()

    @staticmethod
    def get_range_of_units():
        return list(range(0, 100 + 1))

    @staticmethod
    def get_legal_bets(units):
        if units > 50:
            return list(range(1, 100 - units + 1))
        return list(range(1, units + 1))

    def perform_bet(self, bet):
        # print("Bet placed: " + str(bet))
        # print("Flips coin ..")
        flip_result = self.coin.flip()
        if flip_result:
            # print("Right side!")
            return bet
        # print("Wrong side..")
        return -bet


class GamblerWorld:
    def __init__(self):
        self.environment = GamblerEnv()
        self.player = GamblerPlayer(self.environment)

    def get_actions(self):
        return self.player.get_possible_bets()

    def get_state(self):
        return self.player.get_units()

    def get_all_possible_states(self):
        return self.environment.get_range_of_units()

    def get_possible_actions_from_state(self, state):
        # State equals amount of units
        return self.environment.get_legal_bets(int(state))

    def do_action(self, action):
        self.player.place_bet(action)

    def get_reward(self):
        return self.player.get_reward()

    def is_game_over(self):
        state = self.get_state()
        if state == 100:
            # print("You won!")
            return True
        elif state == 0:
            # print("You lost..")
            return True
        return False

    def reset_sim_world(self):
        self.player.set_start_units()

    def save_history(self, episode, str_states):
        pass

    @staticmethod
    def print_end_results(policy):
        # Plot: State vs Bet
        states = list(range(0, 100 + 1))
        best_bets = [0]

        for i in range(1, 100):
            if i in policy.keys():
                best_bets.append(max(policy[i], key=policy[i].get))
            else:
                best_bets.append(0)

        best_bets.append(0)
        print("Best_bets: " + str(best_bets))

        # Plotting the points
        plt.plot(states, best_bets)

        # Name the axis and set title
        plt.xlabel("State")
        plt.ylabel("Bet")
        plt.title("Policy after episodes")
        plt.show()

    def print_episode(self, episode):
        pass

    def save_episode_for_print(self):
        pass
