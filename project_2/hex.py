import numpy as np
import random
import copy

from sklearn import neighbors
from parameters import (
    hex_board_size,
)


class Piece:
    def __init__(self, position, player, is_start_edge, is_end_edge):
        self.position = position  # [x,y]
        self.neighbouring_friends = []  # ex: [Piece1, Piece3]
        self.player = player  # 1 or 2
        self.is_start_edge = is_start_edge  # True/False
        self.is_end_edge = is_end_edge  # True/False
        self.has_visited = False

    def add_neighbouring_friends(self, neighbours):
        if isinstance(neighbours, list):
            # [Piece1, Piece2] += [Piece4] = [Piece1, Piece2, Piece4]
            # [Piece1, Piece2].append([Piece4]) = [Piece1, Piece2, [Piece4]]
            self.neighbouring_friends += neighbours
        else:
            self.neighbouring_friends.append(neighbours)

        # add this object (self) to list of neighbours of all its neighbours
        for neighbour in self.neighbours:
            neighbour.add_neighbouring_friends(self)

    def get_position(self):
        return self.position

    def get_neighbouring_friends(self):
        return self.neighbouring_friends

    def get_player(self):
        return self.player

    def get_is_edge(self):
        return (self.is_start_edge or self.is_end_edge)

    def get_is_start_edge(self):
        return self.is_start_edge

    def get_is_end_edge(self):
        return self.is_end_edge


class Hex:
    def __init__(self):
        self.board = None
        self.current_player = None
        self.neighbours = [(-1, 0), (0, -1), (-1, 1), (0, 1), (1, 0), (1, -1)]
        self.last_move = None
        self.pieces = {
            1: [],
            2: []
        }

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
        # TODO: Figure out how to do when using Piece objects
        """ Reformats the presentation of the board
        for the RL system """
        pass

    def switch_player(self):
        next_player_dict = {
            1: 2,
            2: 1,
        }
        self.current_player = next_player_dict[self.current_player]

    def find_neighbours(self):
        pos = self.last_move.get_position()  # [x, y]
        player = self.last_move.get_player()
        neighbouring_friends = []
        for relative_neighbour in self.neighbours:
            check_pos = []
            check_pos[0] = pos[0] + relative_neighbour[0]
            check_pos[1] = pos[1] + relative_neighbour[1]

            # Get piece on possible neighbour position
            # Piece object in position of neighbour
            possible_neighbour = self.board[check_pos[0]][check_pos[1]]
            neighbouring_player = possible_neighbour.get_player()

            # Add only friendly neighbours (BFFs)
            if neighbouring_player == player:
                neighbouring_friends.append(possible_neighbour)

        return neighbouring_friends

    def is_edge(self, position, player):
        # position = [x][y]
        # player = 1 or 2
        start_edge = False
        end_edge = False
        check = (1 if player == 1 else (0 if player == 2 else None))

        # Player 1 (R) edge: self.board[][0] or self.board[][hex_board_size-1]
        # Player 2 (B) egde: self.board[0][] or self.board[hex_board_size-1][]

        # Check for start edge
        if position[check] == 0:
            start_edge = True

        # Check for end edge
        elif position[check] == hex_board_size-1:
            end_edge = True

        return start_edge, end_edge

    # actual move = [player, piece_positon], ex: [2, [2,1]]
    def perform_move(self, actual_move):
        moving_player = actual_move[0]
        position = actual_move[1]
        row = position[0]
        col = position[1]
        start_edge, end_edge = self.is_edge(position, moving_player)
        # Make a new piece from the state info
        piece = Piece(position, moving_player, start_edge, end_edge)
        self.last_move = piece  # save last move to use in game_over check

        # Find all neighbouring pieces of same player and add them to friendly neighbours in Piece
        friendly_neighbours = self.find_neighbours()
        piece.add_neighbouring_friends(friendly_neighbours)

        # Add piece to board and list of pieces
        self.board[row][col] = piece
        self.pieces[moving_player] = piece

        # Switch to next player
        self.switch_player()

    def find_edge_neighbour(self):

    def game_over(self):
        # Check for full board
        if 0 not in self.board:
            return True

        # Player 1 (R) edge: self.board[][0] or self.board[][hex_board_size-1]
        # Player 2 (B) edge: self.board[0][] or self.board[hex_board_size-1][]

        possible_winner = [1 if self.current_player == 2 else 2]

        # Check for pieces in the edges

        is_start_edge = False
        is_end_edge = False
        for piece in self.pieces[possible_winner]:
            is_start_edge = (piece.is_start_edge()
                             if is_start_edge == False else True)
            is_end_edge = (piece.is_end_edge()
                           if is_start_edge == False else True)

            # if start and end edges exist, break loop
            if is_start_edge and is_end_edge:
                break

        # No pieces on both edges
        if not (is_start_edge and is_end_edge):
            return False

        # Search after winning path from last moved piece position to both start and end edges
        for neighbours in self.last_move.get_neighbouring_friends():
            pass

            # TODO: Continue on this function
            # Use a*?
            # Search from last moved piece and find both outer edges

        else:
            return False

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
        def color_mapping(i): return " RB"[int(i)]
        for r in range(rows):
            row_mid = " "*indent
            row_mid += " {} | ".format(r+1)
            row_mid += " | ".join(map(color_mapping, board[r]))
            row_mid += " | {} ".format(r+1)
            print(row_mid)
            row_bottom = " "*indent
            row_bottom += " "*3+" \\_/"*cols
            if r < rows-1:
                row_bottom += " \\"
            print(row_bottom)
            indent += 2
        headings = " "*(indent-2)+headings
        print(headings)
