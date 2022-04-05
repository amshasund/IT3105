models

## please_model_4x4

hex_board_size = 4    # 3 <= k <= 10
number_actual_games = 1000
print_games = [1, number_actual_games]  # games to be printed out while running
number_search_games = 1000       # test 500
starting_player = 1      # 1 or -1

#mcts parameters
epsilon = 0.9  # eploration constant -> should be decayed? TEST SOMETHING HERE
#episodes = 10
#search_games_per_move = 3
temperature = 0.1
decay_at_action = 1000 # TODO: Which value here??? Gave nan some places when very small 

#actor net
learning_rate = 0.01          # 0.1 is too large and 0.0001 might be too small
hidden_layers = [10, 50, 10]  # 4 layers are a lot more complicated
#["linear", "sigmoid", "tanh", "ReLU"] DO NOT USE LINEAR!!!!
activation_function = ["ReLU", "ReLU", "ReLU"]
optimizer = "adam"  # adagrad, stochastic gradient descent, rmsprop or adam
num_cached = 6
train_interval = 5 # TODO: samkjøre med save interval
epochs = 2
k = 256     # TODO:find better name
batch_size = 256         # default: 32 (2^x)

#tournament
save_interval = number_actual_games // (num_cached-1)
num_agents = number_actual_games // save_interval      # M different agents
games_pr_meet = 50  # G number of games between any two agents in a serie

## please_low_model_4x4

hex_board_size = 4    # 3 <= k <= 10
number_actual_games = 1000
print_games = [1, number_actual_games]  # games to be printed out while running
number_search_games = 1000       # test 500
starting_player = 1      # 1 or -1

#mcts parameters
epsilon = 0.9  # eploration constant -> should be decayed? TEST SOMETHING HERE
#episodes = 10
#search_games_per_move = 3
temperature = 0.1
decay_at_action = 1000 # TODO: Which value here??? Gave nan some places when very small 

#actor net
learning_rate = 0.001          # 0.1 is too large and 0.0001 might be too small
hidden_layers = [10, 50, 10]  # 4 layers are a lot more complicated
#["linear", "sigmoid", "tanh", "ReLU"] DO NOT USE LINEAR!!!!
activation_function = ["ReLU", "ReLU", "ReLU"]
optimizer = "adam"  # adagrad, stochastic gradient descent, rmsprop or adam
num_cached = 6
train_interval = 5 # TODO: samkjøre med save interval
epochs = 2
k = 256     # TODO:find better name
batch_size = 256         # default: 32 (2^x)

#tournament
save_interval = number_actual_games // (num_cached-1)
num_agents = number_actual_games // save_interval      # M different agents
games_pr_meet = 50  # G number of games between any two agents in a serie