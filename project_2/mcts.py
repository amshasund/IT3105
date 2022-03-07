
""" 
# Search to a leaf and update hex_mc
                for search_game in range(number_search_games):
                    is_leaf = False
                    while not is_leaf:
                        leaf, is_leaf = self.mcts.search_to_leaf(root) # TODO: Kan vi sende inn board som en parameter her? 
                        hex_mc.update_leaf(leaf)
                    is_final = False
                    while not is_final:
                        final, is_final = self.anet.choose_rollout(leaf)
                        hex_mc.update_final(final)
                    
                    # Perform Backpropagation 
                    self.mcts.perform_backpropagation(final, root)
                    """