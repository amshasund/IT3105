import copy
import random
from parameters import (
    number_search_games, starting_player
)


class Node:
    def __init__(self, state, parent=None):
        self.parent = parent
        self.children = []
        self.state = state

    def get_state(self):
        return self.state

    def set_parent(self, parent):
        self.parent = parent

    def get_parent(self):
        return self.parent

    def add_children(self, children):
        if isinstance(children, list):
            self.children += children
        else:
            self.children.append(children)

    def get_children(self):
        return self.children


class MonteCarloTree:
    def __init__(self, anet):
        self.root = None
        self.anet = anet

    def init_tree(self, root):
        self.root = Node(root)

    def node_to_leaf(self, node):
        leaf = False
        while not leaf:
            # Node has no children -> node is leaf node
            if len(node.get_children()) == 0:
                leaf = node
            # If node has children -> not a leaf node -> get node's children:
            else:
                # TODO: use tree policy to choose child to look at
                # now: choosing a random child
                node = random.choice(node.get_children())
        return leaf

    def expand_leaf(self, parent, game):
        # get list of all legal moves on current board
        legal_moves = game.get_legal_moves(game.get_hex_board())
        # get player based on who made the previous move
        if game.prev_move:
            player = (2 if game.prev_move.get_player() == 1 else 1)
        else:
            player = starting_player

        for move in legal_moves:
            if move == 1:
                # make a copy of game for simulation
                # TODO: Is this bad????
                temp_game = copy.deepcopy(game)
                # simulate a legal move on the game
                temp_game.perform_move([player, move])
                # get new game state
                state = temp_game.get_state()
                # make child node with new game state and parent
                child = Node(state, parent)
                # add child to parent
                parent.add_children(child)
                # reset temp_game for new loop (not necessary)
                temp_game = None

    def search(self, hex_mc):
        game = hex_mc

        # Search to a leaf and update hex_mc
        for _ in range(number_search_games):
            '''
            1: use tree policy Pt to search from root to a leaf L of MCT - update game with each move

            1.1: expand leaf
            2: use anet to choose rollout actions from L to a final state F - update game with each move
            3: perform mcts backpropagation from F to root
            '''
            # Find a leaf based on tree policy
            leaf = self.node_to_leaf(self.root)
            # Update game with move to leaf node
            game.set_game_state(leaf.get_state())
            # Expand leaf with all its children (legal moves)
            # TODO: Is this just to know which nodes have been rolled out??
            self.expand_leaf(leaf, game)

            # Rollout from leaf with actor network policy
            while not game.game_over():
                move = self.anet.choose_move(
                    game.get_state(True), game.get_legal_moves(game.get_hex_board()))
                game.perform_move(move)

            final = game.get_state()
            # Perform Backpropagation
            self.perform_backpropagation(final, self.root)

    def perform_backpropagation(final, root):
        node = final
        while node is not root:
            node = None          # TODO: Implement

    def select_moce(self):
        pass
