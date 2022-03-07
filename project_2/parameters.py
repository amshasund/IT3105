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

# TODO: FIX THESE

hex_board_size = 3      # 3 <= k <= 10
number_actual_games = 1     # TODO: What is this? Same as num_games?
number_search_games = 1     # TODO: What is this?

# mcts parameters
episodes = 1
search_games_per_move = 1
# etc

# actor net
save_interval = 20 # TODO: what is the range []
learning_rate = 0.1
hidden_layers = [1, 5, 2, 4]
activation_function = ["linear", "sigmoid", "tanh", "RELU"]
optimizer = "sgd"  # adagrad, stochastic gradient descent, rmsprop or adam
num_cached = 3

# round-robin play
num_games = 5
