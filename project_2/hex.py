import random
from parameters import hex_board_size

class Piece:
    def __init__(self, position, player, is_start_edge, is_end_edge):
        self.position = position  # [x,y]
        self.neighbouring_friends = []  # ex: [Piece1, Piece3]
        self.player = player  # 1 or 2
        self.is_start_edge = is_start_edge  # True/False
        self.is_end_edge = is_end_edge  # True/False
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

    def get_is_start_edge(self):
        return self.is_start_edge

    def get_is_end_edge(self):
        return self.is_end_edge

    def is_visited(self):
        return self.visited

    def visit(self, is_visited=True):
        self.visited = is_visited


class Hex:
    def __init__(self):
        self.board = None
        self.current_player = None
        self.last_move = None
        self.edge_pieces = {
            1: [[], []],
            2: [[], []],
        }
        self.is_winner = False

    def init_game_board(self):
        self.board = [[0 for i in range(hex_board_size)]
                      for j in range(hex_board_size)]
        self.current_player = random.randint(1, 2)

    def set_game_state(self, state):
        self.current_player = state[0]
        self.board = state[1]
    
    def get_hex_board(self):
        return self.board

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

    def find_neighbours(self):  # TODO: bedre å ta inn pos og player her??
        pos = self.last_move.get_position()  # [x, y]
        player = self.last_move.get_player()
        neighbouring_friends = []
        relative_neighbours = [
            (-1, 0), (0, -1), (-1, 1), (0, 1), (1, 0), (1, -1)]
        for rel in relative_neighbours:
            check_pos = [pos[0] + rel[0], pos[1] + rel[1]]
            #check_pos.append(pos[0] + rel[0])
            #check_pos.append(pos[1] + rel[1])

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
    
    def get_legal_moves(self, board):
        self.print_game_board(board)
        legal_moves = [(ix,iy) for ix, row in enumerate(board) for iy, i in enumerate(row) if i == 0]
        print("Legal moves: ", legal_moves)
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
        if (row, col) not in legal_moves:
            print("Illegal move registrated - only one of the following moves: " + str(legal_moves))
            return False

        # Check if piece is placed on a edge
        start_edge, end_edge = self.is_edge(position, moving_player)

        # Make a new piece from the state info
        piece = Piece(position, moving_player, start_edge, end_edge)
        self.last_move = piece  # save last move to use in game_over check

        # Find all neighbouring pieces of same player and add them to friendly neighbours in Piece
        friendly_neighbours = self.find_neighbours()
        piece.add_neighbouring_friends(friendly_neighbours)

        # Add piece to board and list of pieces
        self.board[row][col] = piece

        # Add possible start and end pieces to the dictionary
        # to keep track of end pieces for each player
        if start_edge:
            # TODO: Remember to reset these lists after a game
            self.edge_pieces[moving_player][0].append(piece)
        elif end_edge:
            # TODO: Remember to reset these lists after a game
            self.edge_pieces[moving_player][1].append(piece)

        # Switch to next player
        self.switch_player()
    
    def reset_visit(self):
        for row in self.board:
            for cell in row:
                if isinstance(cell, Piece):
                    cell.visit(False)

    def search_path(self, start_piece, end_piece):
        """ Method that suches for a path from start_piece to en_piece
        and returns True if a path is found and False otherwise """
        # TODO: Check for longer path
        start_piece.visit()
        if start_piece.neighbouring_friends:
            #print(start_piece.neighbouring_friends)
            for neighbour in start_piece.neighbouring_friends:
                while neighbour is not end_piece and not neighbour.is_visited():
                    path_exists = self.search_path(neighbour, end_piece)
                    if path_exists:
                        return True
                if neighbour is end_piece:
                    return True
        return False

    def game_over(self):
        player = self.last_move.get_player()
        start_edge = self.edge_pieces[player][0]
        end_edge = self.edge_pieces[player][1]
        #print(start_edge)
        #print(end_edge)

        # Pieces on both edges for current player
        if len(start_edge) >= 1 and len(end_edge) >= 1:
            # Search all possible path combinations
            # for every start piece to every end piece
            for start_piece in start_edge:
                for end_piece in end_edge:
                    path = self.search_path(start_piece, end_piece)
                    self.reset_visit()
                    if path:
                        #TODO: self.reset_board()
                        return True
        else:
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
