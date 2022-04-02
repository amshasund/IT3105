import copy
import random
import numpy as np
from parameters import (
    number_search_games, starting_player
)


class Node:
    def __init__(self, state, action=None, parent=None):
        self.parent = parent
        self.preceding_action = action
        self.children = []
        self.state = state # [player, board.flatten]
        self.count = 0
        self.value = 0

    def get_state(self):
        return self.state
    
    def get_preceding_action(self): # get_move
        return self.preceding_action
    
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

    def search_to_leaf(self):
        node = self.root
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

    def expand_leaf(self, parent, manager, search_game):
        # get list of all legal actions from current state
        legal_actions = manager.get_legal_actions(search_game)
        
        for i in range(len(legal_actions)):
            if legal_actions[i] == 1:
                state, action = manager.try_action(search_game, i)
                child = Node(state, action, parent)
                parent.add_children(child)

    def search(self, manager):
        # Search to a leaf and update hex_mc
        for _ in range(number_search_games):
            '''
            1: use tree policy Pt to search from root to a leaf L of MCT - update game with each move

            1.1: expand leaf
            2: use anet to choose rollout actions from L to a final state F - update game with each move
            3: perform mcts backpropagation from F to root
            '''

            # Find a leaf based on tree policy
            leaf = self.search_to_leaf()

            # Update game with move to leaf node
            search_game = manager.start_game(leaf.get_state())
            
            # Expand leaf with all its children (legal moves)
            self.expand_leaf(leaf, manager, search_game)

            # Rollout from leaf with actor network policy
            start_state = manager.get_state(search_game)

            # ROLLOUT START
            while not manager.is_final(search_game):
                action = self.anet.rollout(
                    manager.get_state(search_game), manager.get_legal_actions(search_game))
                manager.do_action(search_game, action)
            final_state = manager.get_state(search_game)
            # ROLLOUT END

            # Get reward: 1 if outcome is good, -1 if outcome is bad
            reward = manager.get_reward(start_state, final_state)
            
            # Perform Backpropagation
            self.perform_backpropagation(leaf, reward)

    def perform_backpropagation(self, final, reward):
        final.update_count()
        final.update_value(reward)
        if final.get_parent():    
            self.perform_backpropagation(final.get_parent(), reward)
        
    def get_distribution(self, legal_actions):
        dist = np.array(copy.deepcopy(legal_actions))
        # get visit count from all children and place in game
        for child in self.root.get_children():
            index = child.get_preceding_action()
            visit_count = child.get_count()
            dist[index] *= visit_count
        return dist

    def retain_and_discard(self, succ_state):
        # retain subtree rooted at succ_state
        new_root = self.get_node_from_state(succ_state, self.root)
        self.root = new_root

        # discard everything else
        self.root.set_parent(None)

    def get_node_from_state(self, state, node):
        # return node that has state
        if np.array_equal(node.get_state(), state):
            return node
        # if not, search recursively among node's children
        elif node.get_children():
            for child in node.get_children():
                result = self.get_node_from_state(state, child)
                if result:
                    return result
        return None