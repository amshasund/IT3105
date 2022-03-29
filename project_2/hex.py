import random
import numpy as np
from parameters import hex_board_size, starting_player

class Piece:
    def __init__(self, position, player):
        self.position = position  # [x,y]
        self.neighbouring_friends = []  # ex: [Piece1, Piece3]
        self.player = player  # 1 or 2
        self.visited = False  # to use for search

    def add_neighbouring_friends(self, neighbours):
        if isinstance(neighbours, list):
            # add neighbours to list of neighbours
            self.neighbouring_friends = neighbours
            for neighbour in self.neighbouring_friends:
                # add self to neighbours' list of neighbours
                neighbour.add_neighbouring_friends(self)
        else:
            if neighbours not in self.neighbouring_friends:
                self.neighbouring_friends.append(neighbours)

    def get_position(self):
        return self.position

    def get_neighbouring_friends(self):
        return self.neighbouring_friends

    def get_player(self):
        return self.player

    def is_visited(self):
        return self.visited

    def visit(self, is_visited=True):
        self.visited = is_visited


class Hex:
    def __init__(self):
        self.board = None
        self.prev_move = None


    def init_game_board(self):
        self.board = [[0 for i in range(hex_board_size)]
                      for j in range(hex_board_size)]
        self.prev_move = None

    
    def reset_game_board(self):
        self.board = None
        self.prev_move = None

    
    def get_hex_board(self):
        return self.board
    
    def set_game_state(self, state):
        self.board = state[0]
        self.prev_move = state[1]

    def get_state(self, reformat=False):
        if reformat:
            return self.reformat_state()
        return [self.board, self.prev_move]
    
    def get_reward(self, winner, player):
        if winner:
            if winner == player:
                return 1
            else:
                return -1
        else:
            return 0

    def reformat_state(self):
        # TODO: Figure out how to do when using Piece objects
        """ Reformats the presentation of the board
        for the RL system """
        ref_board = [[piece.get_player() if isinstance(piece, Piece) else piece for piece in row] for row in self.board]
        ref_board = np.array(ref_board).flatten()
        if self.prev_move:
            next_player = self.switch_player(self.prev_move.get_player())
        else:
            next_player = starting_player
        return [ref_board, next_player]
        

    def switch_player(self, last_player):
        next_player_dict = {
            1: 2,
            2: 1,
        }
        return next_player_dict[last_player]

    def find_neighbours(self, pos, player): 
        neighbouring_friends = []
        relative_neighbours = [
            (-1, 0), (0, -1), (-1, 1), (0, 1), (1, 0), (1, -1)]
        for rel in relative_neighbours:
            check_pos = [pos[0] + rel[0], pos[1] + rel[1]]

            # Make sure we only look at positions on the board
            if any(n < 0 for n in check_pos) or any(m >= hex_board_size for m in check_pos):
                continue

            # Get piece on possible neighbour position
            # Piece object in position of neighbour
            possible_neighbour = self.board[check_pos[0]][check_pos[1]]

            # Check if neighbour is not a Piece
            if isinstance(possible_neighbour, Piece):
                neighbouring_player = possible_neighbour.get_player()

                # Add only friendly neighbours (BFFs)
                if neighbouring_player == player:
                    neighbouring_friends.append(possible_neighbour)

        return neighbouring_friends
    
    def get_legal_moves(self, board):
        legal_moves = [[1 if i == 0 else 0 for i in row] for row in self.board]
        return legal_moves

    def perform_move(self, actual_move):
        # actual move = [player, piece_positon], ex: [2, [2,1]]
        moving_player = actual_move[0]
        position = actual_move[1]

        # Setting position coordinates
        row = position[0]
        col = position[1]
        
        # Check if actual move is legal
        legal_moves = self.get_legal_moves(self.board)
        if legal_moves[row][col] == 0:
            print("Illegal move registrated")
            return False

        # Make a new piece from the state info
        piece = Piece(position, moving_player)
        self.prev_move = piece  # save last move to use in game_over check

        # Find all neighbouring pieces of same player and add them to friendly neighbours in Piece
        friendly_neighbours = self.find_neighbours(self.prev_move.get_position(), self.prev_move.get_player())
        piece.add_neighbouring_friends(friendly_neighbours)

        # Add piece to board and list of pieces
        self.board[row][col] = piece
        self.print_game_board(self.board)
    
    def reset_visit(self):
        for row in self.board:
            for cell in row:
                if isinstance(cell, Piece):
                    cell.visit(False)

    def search_path(self, start_piece, end_edge):
        """ Method that suches for a path from start_piece to an end_piece
        in end_edges and returns True if a path is found and False otherwise """
        start_piece.visit()
        if start_piece.neighbouring_friends:
            for neighbour in start_piece.neighbouring_friends:
                while neighbour not in end_edge and not neighbour.is_visited():
                    path_exists = self.search_path(neighbour, end_edge)
                    if path_exists:
                        return True
                if neighbour in end_edge:
                    return True
        return False 
    
    def get_edges(self, player):
        if player == 1:
            start = [row[0] for row in self.board if (isinstance(row[0], Piece) and row[0].get_player() == player) ]
            end = [row[-1] for row in self.board if (isinstance(row[-1], Piece) and row[-1].get_player() == player)]
        if player == 2:
            start = [p for p in self.board[0] if (isinstance(p, Piece) and p.get_player() == player)]
            end = [p for p in self.board[-1] if (isinstance(p, Piece) and p.get_player() == player)]
        return start, end

    def game_over(self):
        if self.prev_move:
            player = self.prev_move.get_player()
            start_edge, end_edge = self.get_edges(player)

            # Pieces on both edges for current player
            if len(start_edge) >= 1 and len(end_edge) >= 1:
                # Search possible path combinations from all start pieces
                for start_piece in start_edge:
                    path = self.search_path(start_piece, end_edge)
                    self.reset_visit()
                    if path:
                        return player
        return False

    def print_game_board(self, board):
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

        def color_mapping(i): return " RB"[int(
            i.get_player()) if isinstance(i, Piece) else int(i)]
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
        print("\nR: Player 1 (num) \nB: Player 2 (alph)")
