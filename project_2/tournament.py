from anet import ANet
from hex import StateManager
from parameters import games_pr_meet, number_actual_games, save_interval
from itertools import permutations
import numpy as np
import tensorflow as tf
import absl.logging
from numpy import argmax
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
absl.logging.set_verbosity(absl.logging.ERROR)


class Tournament:
    def __init__(self) -> None:
        self.agents = dict()
        self.series = []
        self.winners = []
        self.manager = StateManager()
        self.anet = ANet()

    def add_agents(self, create_agents=False):
        """Create agents with pretrained models and add to list"""
        for i in range(0, number_actual_games+1, save_interval):
            self.agents[i] = tf.keras.models.load_model(
                "best_models/little_possible_model_4x4_{}.h5".format(i))
        
    def play_games(self):
        for serie in self.series:
            players = dict()
            players[1] = self.agents[serie[0]]
            players[-1] = self.agents[serie[1]]
            # Count winning statistics
            p1_wins = 0
            p2_wins = 0
            print("Playing ", games_pr_meet, "games between model: ", serie[0], " and ", serie[1])
            for i in range(games_pr_meet):
                game = self.manager.start_game()
                while not self.manager.is_final(game):
                    # Get state
                    state = self.manager.get_state(game)
                    legal_actions = self.manager.get_legal_actions(game)
                    best_action = self.anet.choose_action(
                        state, players[state[0]], legal_actions)
                    self.manager.do_action(game, best_action)
                if self.manager.is_final(game) == 1:
                    p1_wins += 1
                elif self.manager.is_final(game) == -1:
                    p2_wins += 1

            self.winners.append((p1_wins, p2_wins))

    def create_series(self):
        # Get all combinations <- TODO: permutations, but not against itself
        self.series = list(permutations(self.agents.keys(), 2))
        

    def set_up_tournament(self):
        # Get competing agents
        self.add_agents()

        # Create series
        self.create_series()

    def play_tournament(self):
        # Set up tournament
        self.set_up_tournament()
        print("----------------- TOURNAMENT -----------------")
        
        # Play
        self.play_games()
        wins = dict()
        for i in range(len(self.series)):
            home = self.series[i][0]
            away = self.series[i][1]
            wins[home] = (0 if not home in wins.keys()
                          else wins[home]) + self.winners[i][0]
            wins[away] = (0 if not away in wins.keys()
                          else wins[away]) + self.winners[i][1]
        print(wins)


def play_model_against_random(model):
    print("-----------------Play model agains random -----------------")
    anet = ANet()
    anet.build_model()

    for _ in range(0, 10):
        manager = StateManager()
        winners = []
        
        # Play games
        for i in range(100):
            game = manager.start_game()
            
            while not manager.is_final(game):
                state = manager.get_state(game)
                if manager.get_next_player(state) == 1:
                    # Network model
                    state_init = manager.get_state(game)
                    legal_actions = manager.get_legal_actions(game)
                    best_action = anet.choose_action(
                        state_init, model, legal_actions)
                    manager.do_action(game, best_action)
                else:
                    # Random
                    action = random.choice(np.argwhere(
                        manager.get_legal_actions(game) == 1).reshape(-1))
                    manager.do_action(game, action)
                    
            winner = manager.is_final(game)
            assert winner is not False
            winners.append(winner)
        print("wins: ", sum(v==1 for v in winners), " of 100 games")
