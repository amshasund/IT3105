import copy
import random
import numpy as np
from parameters import (
    number_search_games, starting_player
)


class Node:
    def __init__(self, state, move=None, parent=None):
        self.parent = parent
        self.move = move
        self.children = []
        self.state = state
        self.count = 0
        self.value = 0

    def get_state(self):
        return self.state
    
    def get_board(self):
        return self.state[0]
    
    def get_move(self):
        return self.move
    
    def get_non_object_board(self):
        return [[piece.get_player() if not isinstance(piece, int) else piece for piece in row] for row in self.state[0]]
    
    def get_player(self):
        return (self.state[1].get_player() if self.state[1] else starting_player)
    
    def get_count(self):
        return self.count
    
    def get_value(self):
        return self.value
    
    def update_count(self):
        self.count += 1
    
    def update_value(self, value):
        # TODO: Do maths here!!
        self.value += value

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
    
    def get_root(self):
        return self.root

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
        legal_moves = game.get_legal_moves()
        # get player based on who made the previous move
        player = game.get_next_player()
        
        for row in range(len(legal_moves)):
            for col in range(len(legal_moves[row])):
                if legal_moves[row][col] == 1:
                    # make a copy of game for simulation
                    # TODO: Is this bad????
                    temp_game = copy.deepcopy(game)
                    # simulate a legal move on the game
                    temp_game.perform_move([player, [row, col]])
                    # get new game state
                    state = copy.deepcopy(temp_game.get_state())
                    # make child node with new game state and parent
                    move = [row, col]
                    child = Node(state, move, parent)
                    # add child to parent
                    parent.add_children(child)
                    # reset temp_game for new loop (not necessary)
                    temp_game = None

    def search(self, hex_mc, root):
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
            leaf = self.node_to_leaf(root)
            # Update game with move to leaf node
            game.reset_game_board()
            game.set_game_state(copy.deepcopy(leaf.get_state()))
            player = leaf.get_player()
            
            # Expand leaf with all its children (legal moves)
            self.expand_leaf(leaf, game)

            # Rollout from leaf with actor network policy

            # ROLLOUT START
            while not game.game_over():
                move = self.anet.rollout(
                    game.get_state(True), game.get_legal_moves())
                game.perform_move(move)
            winner = game.game_over()
            # ROLLOUT END
            reward = game.get_reward(winner, player)
            
            # Perform Backpropagation
            self.perform_backpropagation(leaf, reward)

    def perform_backpropagation(self, final, reward):
        if final.get_parent():
            final.update_count()
            final.update_value(reward)
            #print("Has parent: ", final.get_board())
            #print("Count: ", final.get_count())
            #print("Value: ", final.get_value())
            self.perform_backpropagation(final.get_parent(), reward)
        else:
            final.update_count()
            final.update_value(reward)
            #print("Has no parent: ", final.get_board())
            #print("Count: ", final.get_count())
            #print("Value: ", final.get_value())
        
    def get_distribution(self, root, legal_moves):
        dist = np.array(copy.deepcopy(legal_moves))
        # get visit count from all children and place in game
        for child in root.get_children():
            index = child.get_move()
            row = index[0]
            col = index[1]
            visit_count = child.get_count()
            dist[row][col] *= visit_count
        #print("Distribution: ", dist)
        return dist

    def retain_and_discard(self, succ_state):
        # retain subtree rooted at succ_state
        #print("Successor state: ", succ_state)
        new_root = self.get_node_from_state(succ_state, self.root)
        #print("New root: ", new_root)
        self.root = new_root

        # discard everything else
        self.root.set_parent(None)

    def get_node_from_state(self, state, node):
        #print("Get node from board: ", state)
        #print("Current node: ", node)
        #print("Board of node: ", node.get_non_object_board())
        # return node that has state
        if node.get_non_object_board() == state:
            #print("Found node: ", node)
            return node
        # if not, search recursively among node's children
        elif node.get_children():
            for child in node.get_children():
                #print("Testing child ", child, " of node ", node)
                result = self.get_node_from_state(state, child)
                if result:
                    return result
        #print("Found nothing!")
        return None