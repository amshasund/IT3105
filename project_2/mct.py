import copy
import random
import numpy as np
from parameters import (
    number_search_games,
    epsilon
)


class Node:
    def __init__(self, state, action=None, parent=None):
        self.parent = parent
        self.preceding_action = action
        self.children = []
        self.state = state  # [player, board.flatten]
        self.count = 1
        self.value = 0

    def get_state(self):
        return self.state

    def get_preceding_action(self):  
        return self.preceding_action

    def get_count(self):
        return self.count

    def get_value(self):
        return self.value

    def update_count(self):
        self.count += 1

    def update_value(self, value):
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
            # Leaf: Node with no children 
            if len(node.get_children()) == 0:
                leaf = node
            else:
                # Calculate tree policy score for all children on current node
                best_child = None
                best_score = -np.inf 
                
                # Go through node's children
                for child in node.get_children():
                    # Check for maximize or minizime player
                    score = child.get_value() + epsilon * \
                        np.sqrt(np.log(node.get_count())/child.get_count())
                    # Check if score is better than the best registrated
                    if score > best_score:
                        best_child = child
                        best_score = score
                # Set a new node whoch is the best child from the old node
                node = best_child
        return leaf

    def expand_leaf(self, parent, manager, search_game):
        # Get list of all legal actions from current state
        legal_actions = manager.get_legal_actions(search_game)

        for i in range(len(legal_actions)):
            if legal_actions[i] == 1:
                state, action = manager.try_action(search_game, i)
                child = Node(state, action, parent)
                parent.add_children(child)

    def search(self, manager, model):
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

            # Check if leaf is final state
            if manager.is_final(search_game):
                reward = manager.get_reward(start_state, start_state)
                self.perform_backpropagation(leaf, reward)
                return

            # ROLLOUT START
            # Get first action to know which child of leaf is chosen
            first_action = self.anet.choose_action(manager.get_state(
                search_game), model, manager.get_legal_actions(search_game))
            # Do first ation
            manager.do_action(search_game, first_action)
            # Play until game over
            while not manager.is_final(search_game):
                # Get action
                action = self.anet.choose_action(manager.get_state(
                    search_game), model, manager.get_legal_actions(search_game))
                # Do action
                manager.do_action(search_game, action)
            final_state = manager.get_state(search_game)
            # ROLLOUT END

            # Get reward: 1 if outcome is good, -1 if outcome is bad
            reward = manager.get_reward(start_state, final_state)

            # Perform Backpropagation
            child = next(child for child in leaf.get_children()
                         if child.preceding_action == first_action)
            self.perform_backpropagation(child, reward)

    def perform_backpropagation(self, final, reward):
        # N(s, a)
        final.update_count()
        # Q(s, a)
        final.update_value(reward)
        if final.get_parent():
            # Switch player in order to maximize action for every player
            self.perform_backpropagation(final.get_parent(), -reward)

    def get_distribution(self, legal_actions):
        dist = np.array(copy.deepcopy(legal_actions))
        # Get visit count from all children and place in game
        for child in self.root.get_children():
            index = child.get_preceding_action()
            visit_count = child.get_count()
            dist[index] *= visit_count
        return dist

    def retain_and_discard(self, succ_state):
        # Retain subtree rooted at succ_state and discard everything else
        self.root = Node(succ_state)

    def get_node_from_state(self, state, node):
        # Only return node that has a state
        if np.array_equal(node.get_state(), state):
            return node
        # Otherwise, search recursively among the node's children
        elif node.get_children():
            for child in node.get_children():
                result = self.get_node_from_state(state, child)
                if result:
                    return result
        return None
