from parameters import (
    number_search_games
)

class Node:
    def __init__(self, state):
        self.parent = None
        self.children = []
        self.state = state
    
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
    def __init__(self) -> None:
        self.root  = None
    
    def init_tree(self, root):
        self.root = Node(root)

    def search(self, game):
        # Search to a leaf and update hex_mc
        for search_game in range(number_search_games):
            is_leaf = False
            # loop start at node
            # know which node is the current node
            # dybde først - første barnet? eller random? tree policyen bestemmer hvilekt barn man skal besøke
            # leaf = så lenge noden ikke har barn
            # tree policy : hvordan søke ned til leaf slides prosjekt 2 - tree policy - monte carlo tree search
            while not is_leaf:
                leaf, is_leaf = self.search_to_leaf(root) # TODO: Kan vi sende inn board som en parameter her? 
                game.update_leaf(leaf)
            is_final = False
            while not is_final:
                final, is_final = self.anet.choose_rollout(leaf)
                game.update_final(final)
            
            # Perform Backpropagation 
            self.perform_backpropagation(final, root)
    
    def perform_backpropagation(final, root):
        pass