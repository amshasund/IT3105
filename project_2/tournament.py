from numpy import argmax
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import tensorflow as tf
import numpy as np
from itertools import permutations
from parameters import games_pr_meet, number_actual_games, save_interval
from hex import StateManager
from anet import ANet

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
            #self.agents[i]= tf.keras.models.load_model("model34x4_{}.h5".format(i))
            self.agents[i]= tf.keras.models.load_model("lite_model3x3_{}.h5".format(i))
    
    def play_games(self):
        for serie in self.series:
            players = dict()
            players[1] = self.agents[serie[0]]
            players[2] = self.agents[serie[1]]
            # Count winning statistics
            p1_wins = 0
            p2_wins = 0
            for i in range(games_pr_meet):
                game = self.manager.start_game()
                while not self.manager.is_final(game):
                    # Get state
                    state = self.manager.get_state(game)
                    legal_actions = self.manager.get_legal_actions(game)
                    best_action = self.anet.choose_action(state, players[state[0]], legal_actions)
                    self.manager.do_action(game, best_action)
                if self.manager.is_final(game) == 1:
                    p1_wins += 1
                elif self.manager.is_final(game) == 2:
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
        
        # Play
        self.play_games()
        #print("Series ", self.series[i], " Winners: ", self.winners[i])
        '''
        Series: [(0,1),(0,2),(1,0),(1,2),(2,0),(2,1)]
        Winners: [(0,1),(0,2),(1,0),(1,2),(2,0),(2,1)]
        Matrix
                0           1           2       (home)
         0      x        (22, 3)

         1    (19,6)        x

         2    (13,12)                         x  
        '''
        matrix = [[0 for i in range(len(self.agents))]
                      for j in range(len(self.agents))]
        wins = dict()
        for i in range(len(self.series)):
            home = self.series[i][0]
            away = self.series[i][1]
            wins[home] = (0 if not home in wins.keys() else wins[home]) + self.winners[i][0]
            wins[away] = (0 if not away in wins.keys() else wins[away]) + self.winners[i][1]
            #matrix[home][away] = self.winners[i]
        
        print(wins)
        #print(matrix)
