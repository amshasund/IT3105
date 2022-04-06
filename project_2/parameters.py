"""
• The size (k) of the k x k Hex board, where 3 ≤ k ≤ 10.
• Standard MCTS parameters, such as the number of episodes, number of search games per actual move, etc.
• In the ANET, the learning rate, the number of hidden layers and neurons per layer, along with any of the
following activation functions for hidden nodes: linear, sigmoid, tanh, RELU.
• The optimizer in the ANET, with (at least) the following options all available: Adagrad, Stochastic Gradient
Descent (SGD), RMSProp, and Adam.
• The number (M) of ANETs to be cached in preparation for a TOPP. These should be cached, starting with an
untrained net prior to episode 1, at a fixed interval throughout the training episodes.
• The number of games, G, to be played between any two ANET-based agents that meet during the round-robin
play of the TOPP.
"""
# GENERAL PARAMETERS
number_actual_games = 600
print_games = [1, number_actual_games]  # games to be printed out while running

# HEX
hex_board_size = 4  
starting_player = 1     # 1 or -1

# MCTS
epsilon = 0.9  # eploration constant 
number_search_games = 600  
temperature = 0.01      # For "one-hot-encoding"
decay_at_action = 1000  

# ACTOR NEURAL NET
learning_rate = 0.0001          
hidden_layers = [256, 256]  
activation_function = ["ReLU", "ReLU"]
optimizer = "adam"  
num_cached = 5
train_interval = 5 
epochs = 10  
k = 256     
batch_size = 256      

# TOURNAMENT
save_interval = number_actual_games // (num_cached-1)
num_agents = number_actual_games // save_interval     
games_pr_meet = 25 
