# Select game
game = 'towers_of_hanoi'  # 'the_gambler', 'towers_of_hanoi' or 'pole_balancing'

# pivotal parameters

episodes = 200
max_steps = 300
critic_type = "table"  # "NN" or "table"
neural_dim = [9]
lr_actor = 0.9  # alpha_a
lr_critic = 0.3  # alpha_c
eligibility_decay_actor = 0.9  # lambda_a
eligibility_decay_critic = 0.9  # lambda_c
discount_factor_actor = 0.9  # gamma_a
discount_factor_critic = 0.9  # gamma_c
epsilon = 0.9
display_variable = episodes - 1
frame_delay = 3

# pole balancing

pole_length = 0.5  # [0.1, 1.0] m
pole_mass = 0.1  # [0.05, 0.5] kg
gravity = -9.81  # [-15, -5] m/s^2
timestep = 0.02  # [0.01, 0.1] s

# towers of hanoi

pegs = 3  # [3,5]
discs = 4  # [2,6]

# the gambler

win_probability = 0.6  # [0, 1.0]
