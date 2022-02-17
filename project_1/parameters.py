# Select game
game = 'towers_of_hanoi'  # 'the_gambler', 'towers_of_hanoi' or 'pole_balancing'

# Pivotal parameters
episodes = 200  # The gambler: 10.000
max_steps = 300
critic_type = "NN"  # "NN" or "table"
neural_dim = [9, 5, 1]
lr_actor = 0.01  # alpha_a
lr_critic = 0.1  # alpha_c
eligibility_decay_actor = 0.9  # lambda_a
eligibility_decay_critic = 0.9  # lambda_c
discount_factor_actor = 0.99  # gamma_a
discount_factor_critic = 0.99  # gamma_c
epsilon = 0.9
display_variable = episodes - 1
frame_delay = 1

# Pole balancing
pole_length = 0.5  # [0.1, 1.0] m
pole_mass = 0.1  # [0.05, 0.5] kg
gravity = -9.81  # [-15, -5] m/s^2
timestep = 0.01  # [0.01, 0.1] s

# Towers of hanoi
pegs = 3  # [3,5]
discs = 4  # [2,6]

# The gambler
win_probability = 0.4  # [0, 1.0]
