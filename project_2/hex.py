import random
import numpy as np
import copy
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
        self.next_player = starting_player


    def init_game_board(self):
        self.board = [[0 for i in range(hex_board_size)]
                      for j in range(hex_board_size)]
        self.next_player = starting_player
        #self.next_player = random.randint(1,2)
    
    def reset_game_board(self):
        self.board = None
        self.next_player = starting_player
        #self.next_player = random.randint(1,2)

    def get_hex_board(self):
        return self.board

    def get_next_player(self):
        return self.next_player
    
    def set_game_state(self, state):
        """Creates a new board for simulation with pieces as a copy of the state
        given"""
        # state = [2 0 0 0 1 2 1 0 0 0]
        # first num is player, rest is board
        self.next_player = state[0]
        state = np.delete(state, 0)
        # state = [0 0 0 1 2 1 0 0 0]
        for i in range(len(state)):
            if state[i] != 0:
                pos = list(np.unravel_index(i, np.array(self.board).shape))
                # pos = [x, y]
                
                piece = Piece(pos, state[i])
                friendly_neighbours = self.find_neighbours(pos, self.next_player)
                piece.add_neighbouring_friends(friendly_neighbours)

                row = pos[0]
                col = pos[1]
                self.board[row][col] = piece
    
    def get_reward(self, winner, player):
        # TODO : FIX
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
        ref_board = self.get_non_object_board(self.board)
        ref_board = np.array(ref_board).flatten()
        next_player = self.get_next_player()
        return [ref_board, next_player]

    def get_non_object_board(self):
        return [[piece.get_player() if isinstance(piece, Piece) else piece for piece in row] for row in self.board]
        

    def switch_player(self, prev_player):
        next_player_dict = {
            1: 2,
            2: 1,
        }
        return next_player_dict[prev_player]

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
    
    def get_legal_moves(self):
        legal_moves = [[1 if i == 0 else 0 for i in row] for row in self.board]
        return legal_moves

    def perform_move(self, actual_move, print=False):
        # actual move = [player, piece_positon], ex: [2, [2,1]]
        moving_player = actual_move[0]
        position = actual_move[1]

        # Setting position coordinates
        row = position[0]
        col = position[1]
        
        # Check if actual move is legal
        legal_moves = self.get_legal_moves()
        if legal_moves[row][col] == 0:
            print("Illegal move registrated")
            return False

        # Make a new piece from the state info
        piece = Piece(position, moving_player)

        # Find all neighbouring pieces of same player and add them to friendly neighbours in Piece
        friendly_neighbours = self.find_neighbours(position, moving_player)
        piece.add_neighbouring_friends(friendly_neighbours)

        # Add piece to board and list of pieces
        self.board[row][col] = piece

        self.next_player = self.switch_player(moving_player)

        if print:
            self.print_game_board()
    
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
        next_player = self.get_next_player()
        potential_winner = self.switch_player(next_player)
        start_edge, end_edge = self.get_edges(potential_winner)

        # Pieces on both edges for current player
        if len(start_edge) >= 1 and len(end_edge) >= 1:
            # Search possible path combinations from all start pieces
            for start_piece in start_edge:
                path = self.search_path(start_piece, end_edge)
                self.reset_visit()
                if path:
                    return potential_winner 
        if not any(0 in row for row in self.board):
            return -1                
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


class StateManager:
    
    def start_game(self, specified_state=False):
        game = Hex()
        game.init_game_board()
        if specified_state is not False:
            # TODO: Need to reformat this
            game.set_game_state(specified_state)
        return game

    def get_state(self, game):
        board = np.array(game.get_non_object_board()).flatten()
        player = game.get_next_player()
        state = np.insert(board, 0, player)
        return state
    
    def get_next_player(self, state):
        return state[0]
    
    def do_action(self, game, action, print=False):
        action = np.unravel_index(action, np.array(game.get_hex_board()).shape)
        actual_move = [game.get_next_player(), action]
        game.perform_move(actual_move, print)
    
    def try_action(self, game, action):
        # action is a number in flattened legal action list
        # make a temporary copy of the game
        temp_game = copy.deepcopy(game)

        # simulate a legal move on the temp game
        self.do_action(temp_game, action)
        
        # get new temp game state
        state = copy.deepcopy(self.get_state(temp_game))

        return state, action
    
    def get_legal_actions(self, game):
        return np.array(game.get_legal_moves()).flatten()

    def is_final(self, game):
        return game.game_over()

    def print_state(self, game):
        game.print_game_board()
    
    def get_reward(self, start, final):
        winner = final[0]
        player = start[0]
        if winner:
            if winner == player:
                return 1
            return -1
        return 0

