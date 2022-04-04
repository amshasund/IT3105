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

hex_board_size = 3     # 3 <= k <= 10
number_actual_games = 250
print_games = [1, number_actual_games]  # games to be printed out while running
number_search_games = 1000       # test 500
starting_player = 1      # 1 or -1

# mcts parameters
epsilon = 0.9  # eploration constant -> should be decayed? TEST SOMETHING HERE
#episodes = 10
#search_games_per_move = 3
# etc
temperature = 0.1
decay_at_action = 10

# actor net
learning_rate = 0.02          # 0.1 is too large and 0.0001 might be too small
hidden_layers = [6, 18, 6]  # 4 layers are a lot more complicated
# ["linear", "sigmoid", "tanh", "ReLU"] DO NOT USE LINEAR!!!!
activation_function = ["ReLU", "ReLU", "ReLU"]
optimizer = "adam"  # adagrad, stochastic gradient descent, rmsprop or adam
num_cached = 3
train_interval = 5
batch_size = 32         # default: 32

# Tournament
save_interval = number_actual_games // (num_cached-1)
num_agents = number_actual_games // save_interval      # M different agents
games_pr_meet = 25  # G number of games between any two agents in a serie


# OHT
# 7x7 board
# train beforehand
