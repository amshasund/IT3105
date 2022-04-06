# From pseudocode
from anet import ANet, LiteModel
from mct import MonteCarloTree
from hex import StateManager
import numpy as np
import random

from parameters import (
    number_actual_games,
    save_interval,
    print_games,
    train_interval,
    temperature,
    decay_at_action,
    k
)


class RLSystem:
    def __init__(self):
        self.manager = StateManager()
        self.anet = ANet()
        self.lite_model = LiteModel
        self.mct = MonteCarloTree(self.anet)

    def algorithm(self):
        # Intialize replay buffer
        replay_buffer = [] 

        # Create neural net
        self.anet.build_model()

        # Save untrained net
        self.anet.save_model(0)

        # Play games
        for actual_game in range(1, number_actual_games+1):
            print("Running game: ", actual_game)
            
            # Initialize actual game board
            game = self.manager.start_game()

            # Print first game
            print_game = (True if actual_game in print_games else False)
            if print_game:
                print("START GAME \n")
                self.manager.print_state(game)
            
            # Get start state and initialize a Monte Carlo tree 
            # with the state as the root
            state_init = self.manager.get_state(game)
            self.mct.init_tree(state_init)

            # Play until game over
            action_counter = 0
            while not self.manager.is_final(game):
                # Transform the network model to a lite model
                # for faster running
                lite_model = self.lite_model.from_keras_model(self.anet.model)
                
                # Perform Monte Carlo Tree Search 
                self.mct.search(self.manager, lite_model)
                
                # Get distribution of visit counts in Monte Carlo Tree
                visit_dist = self.mct.get_distribution(
                    self.manager.get_legal_actions(game))
                # Save state and it's visit distribution into a buffer
                # in order to learn from experience
                replay_buffer.append(
                    (self.manager.get_state(game), visit_dist))

                # Normalize visit distribution
                visit_dist = visit_dist/sum(visit_dist)
                
                # Choose actual move from normalized visit distribution
                action = np.random.choice(
                    range(len(visit_dist)), p=visit_dist)

                # Perform move on the game board
                self.manager.do_action(game, action, print=print_game)
                action_counter += 1
                
                # Save the produced successor state
                successor_state = self.manager.get_state(game)
                self.mct.retain_and_discard(successor_state)
            
            # Print the final game
            if print_game:
                print("WINNER: Player", self.manager.is_final(game))

            # Train and save
            if actual_game % save_interval == 0:
                # Train the network model on content stored in the replay buffer
                self.anet.train_model(replay_buffer)
                # Save the replay buffer to a file go generate a dataset
                #self.write_rbuf_to_file(replay_buffer)
                replay_buffer = replay_buffer[-k:]
                # Save the actor net model
                self.anet.save_model(actual_game)
            
    def write_rbuf_to_file(self, rbuf):
        output_file = open('crazy_3x3.txt', 'w')

        for element in rbuf:
            state = list(element[0])
            target = list(element[1])
            output_file.write(str((state, target)) + '\n')

        output_file.close()

    def get_action_dist(self, visit_dist, action_counter):
        visit_dist = visit_dist / sum(visit_dist)
        action_dist = visit_dist**(
            1/(temperature if action_counter < decay_at_action else temperature**10))
        action_dist = action_dist / sum(action_dist)
        return action_dist
