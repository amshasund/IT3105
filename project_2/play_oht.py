from anet import ANet
import tensorflow as tf
import numpy as np

# Import and initialize your own actor
class MyHexActor:
    def __init__(self):
        self.anet = ANet()
        self.agent = tf.keras.models.load_model("oht_models/crazy_bitch_7x7_100.h5")

    def get_action(self, state):
        # state = [
        #   1,          current player
        #   0, 0, 0,    first row
        #   0, 2, 1,    second row
        #   2, 0, 1     ...
        # ]
        # Player 1 goes "top-down" and player 2 goes "left-right"

        # For us it is the opposite:
        # Player 1 goes "left-right" and player -1 goes "top-down"
        # Need to switch
        player = -1 if state[0] == 1 else (1 if state[0] == 2 else 0)
        board = [-1 if i == 1 else (1 if i == 2 else 0) for i in state[1:]]  # flattened
        # Adds correct player to flattened board
        corrected_state = [player] + board
        # Get legal actions from board
        legal = self.get_legal_actions(board)
        # Get flattened action from model
        flat_action = self.anet.choose_action(
                        corrected_state, self.agent, legal)
        # Make flat index into tuple based on board size
        # Get row and col
        row, col = np.unravel_index(flat_action, (7,7))
        return int(row), int(col)
    
    def get_legal_actions(self, board):
        return [1 if i == 0 else 0 for i in board]

actor = MyHexActor()

# Import and override the `handle_get_action` hook in ActorClient
from ActorClient import ActorClient
class MyClient(ActorClient):
    def handle_get_action(self, state):
        row, col = actor.get_action(state) # Your logic
        return row, col

# Initialize and run your overridden client when the script is executed
if __name__ == '__main__':
    client = MyClient()
    client.run()

    # DEBUGGING
    # state = [1, 0, 2, 1, 0, 0, 0, 1, 2, 0]
    # row, col = actor.get_action(state)
    # print(row, col)