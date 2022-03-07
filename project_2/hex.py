import numpy as np
from parameters import (
    hex_board_size,
)

class Hex:
    def __init__(self) -> None:
        self.board = None
    
    def init_game_board(self):
        self.board = np.zeros((hex_board_size, hex_board_size))
        print(self.board)