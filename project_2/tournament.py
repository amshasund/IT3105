from numpy import argmax
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
from itertools import combinations
from parameters import games_pr_meet, number_actual_games, save_interval
from hex import StateManager

class Tournament:
    def __init__(self) -> None:
        self.agents = dict()
        self.series = []
        self.winners = []
        self.manager = StateManager()

    def add_agents(self, create_agents=False):
        """Create agents with pretrained models and add to list"""
        self.agents[125] = tf.keras.models.load_model("super_model_125.h5")
        self.agents[66] = tf.keras.models.load_model("4_model_66.h5")
        
        #self.agents[250] = tf.keras.models.load_model("super_model_250.h5")
        #for i in range(0, number_actual_games+1, save_interval):
            #self.agents[i]= tf.keras.models.load_model("project_2/super_model_{}.h5".format(i))
            #self.agents[i]= tf.keras.models.load_model("4_model_{}.h5".format(i))
    
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
                    best_action = self.get_best_action(state, players[state[0]], legal_actions)
                    self.manager.do_action(game, best_action)
                if self.manager.is_final(game) == 1:
                    p1_wins += 1
                elif self.manager.is_final(game) == 2:
                    p2_wins += 1
            
            self.winners.append((p1_wins, p2_wins))

    def get_best_action(self, state, model, legal_actions):
        state = np.array(state).reshape([1, -1])
        action_dist = np.array(model(state)[0])
        # Remove illegal actions
        action_dist *= legal_actions
        best_action = argmax(action_dist)
        return best_action
        

    def create_series(self):
        # Get all combinations
        self.series = list(combinations(self.agents.keys(), 2))
    
    def set_up_tournament(self):
        # Get competing agents
        self.add_agents()
        
        # Create series
        self.create_series()

    def play_tournament(self):
        # Set up tournament 
        self.set_up_tournament()
        print("Series ", self.series)
        
        # Play
        self.play_games()
        print("Winners: ", self.winners)