import numpy as np
import random
from parameters import (
    hex_board_size,
)

class Hex:
    def __init__(self):
        self.board = None
        self.current_player = None
        self.neighbours = [(-1,0), (0,-1), (-1,1), (0,1), (1,0), (1,-1)]
    
    def init_game_board(self):
        self.board = np.zeros((hex_board_size, hex_board_size))
        self.current_player = random.randint(1, 2)
    
    def set_game_state(self, state):
        self.current_player = state[0]
        self.board = state[1]

    def get_state(self, reformat=False):
        if reformat:
            return [self.current_player, self.reformat_board()]
        return [self.current_player, self.board]
    
    def reformat_board(self):
        """ Reformats the presentation of the board
        for the RL system """
        list_of_pieces = {
            "player1": [],
            "player2": []
        }
        list_of_pieces["player1"] = np.where(self.board == 1)
        list_of_pieces["player2"] = np.where(self.board == 2)
        
        return list_of_pieces

    def print_game_board(self):
        """ Prints a beautiful representation of the Hex board"""
        board = self.board
        column_names = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        rows = len(board)
        cols = len(board[0])
        indent = 0
        headings = " "*5+(" "*3).join(column_names[:cols])
        print(headings)
        tops = " "*5+(" "*3).join("-"*cols)
        print(tops)
        roof = " "*4+"/ \\"+"_/ \\"*(cols-1)
        print(roof)
        color_mapping = lambda i : " RB"[int(i)]
        for r in range(rows):
            row_mid = " "*indent
            row_mid += " {} | ".format(r+1)
            row_mid += " | ".join(map(color_mapping,board[r]))
            row_mid += " | {} ".format(r+1)
            print(row_mid)
            row_bottom = " "*indent
            row_bottom += " "*3+" \\_/"*cols
            if r<rows-1:
                row_bottom += " \\"
            print(row_bottom)
            indent += 2
        headings = " "*(indent-2)+headings
        print(headings)

    def switch_player(self):
        next_player_dict = {
            1 : 2,
            2 : 1,
        }
        self.current_player = next_player_dict[self.current_player]

    def perform_move(self, actual_move):
        moving_player = actual_move[0]
        position = actual_move[1]
        row = position[0]
        col = position[1]
        self.board[row, col] = moving_player
        self.switch_player()

    def game_over(self):
        # Check for full board
        if 0 not in self.board:
            return True
        
        # Check for winner
        # TODO: what to do here?? Use a*?
        # Search from last moved piece and find both outer edges
        
        else:
            return False