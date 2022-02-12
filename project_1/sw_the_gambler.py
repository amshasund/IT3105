import random
import numpy as np

from parameters import win_probability, episodes
import matplotlib.pyplot as plt


class Coin:
    def __init__(self):
        self.win_probability = win_probability

    def flip(self):
        return random.choices(
            population=[False, True],
            weights=[1 - win_probability, win_probability],
            k=1,
        )[0]


class GamblerPlayer:
    def __init__(self, env):
        self.env = env
        self.units = random.randint(1, 99)
        self.reward = 0

    def get_units(self):
        return self.units

    def get_reward(self):
        # For winning
        if self.units == 100:
            self.reward = 100

        # For loosing
        elif self.units == 0:
            self.reward = -100

        # For moving
        else:
            self.reward = (
                -1
            )  # økte antall minus per steg # sjekke antall steg økte antall episoder

        return self.reward

    def set_start_units(self):
        self.units = random.randint(1, 99)

    def place_bet(self, bet):
        # self.reward = self.env.perform_bet(bet)
        self.units += self.env.perform_bet(bet)

    def get_possible_bets(self):
        return self.env.get_legal_bets(self.units)


class GamblerEnv:
    def __init__(self):
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
        # in this game, state equals amount of units
        return self.environment.get_legal_bets(state)

    def do_action(self, action):
        self.player.place_bet(action)

    def get_reward(self):
        return self.player.get_reward()

    def is_game_over(self):
        state = self.get_state()
        if state == 100:
            # print("You won!")
            # Reset number of units for new game
            self.player.set_start_units()
            return True
        elif state == 0:
            # print("You lost..")
            # Reset number of units for new game
            self.player.set_start_units()
            return True
        return False

    @staticmethod
    def print_results(policy):
        states = list(range(0, 100+1))
        best_bets = [0]

        for i in range(1, 100):
            if i in policy.keys():
                best_bets.append(max(policy[i], key=policy[i].get))
            else:
                best_bets.append(0)

        best_bets.append(0)
        # print("Best_bets: " + str(best_bets))

        # Plotting the points
        plt.plot(states, best_bets)

        # Name the axis and set title
        plt.xlabel("State")
        plt.ylabel("Bet")
        plt.title("Policy after episodes")
        plt.show()
