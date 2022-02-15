# Select game
game = 'the_gambler'  # 'the_gambler', 'towers_of_hanoi' or 'pole_balancing'

# pivotal parameters

episodes = 10000  # The gambler: 10.000
max_steps = 300
critic_type = "table"  # "NN" or "table"
neural_dim = [9]
lr_actor = 0.01  # alpha_a       # reduser veldig typ 0.1 eller 0.01
lr_critic = 0.1  # alpha_c      # lavere -> 0.1
eligibility_decay_actor = 0.9  # lambda_a
eligibility_decay_critic = 0.9  # lambda_c
discount_factor_actor = 0.99  # gamma_a
discount_factor_critic = 0.99  # gamma_c
epsilon = 0.9
display_variable = episodes - 1
frame_delay = 1

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
