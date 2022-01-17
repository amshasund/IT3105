# pivotal parameters

episodes = 500
max_steps = 300
critic_type = "neural"                  # "neural or "table"
neural_dim = (15, 20, 30, 5, 1)
lr_actor = 0.9
lr_critic = 0.9
eligibility_decay_actor = 0.9
eligibility_decay_critic = 0.9
discount_factor_actor = 0.5
discount_factor_critic = 0.5
epsilon = 0
display_variable = episodes - 1
frame_delay = 3

# pole balancing

pole_length = 0.5                       # [0.1, 1.0] m
pole_mass = 0.1                         # [0.05, 0.5] kg
gravity = 9.81                          # [5, 15] m/s^2
timestep = 0.02                         # [0.01, 0.1] s

# towers of hanoi

pegs = 3                                # [3,5]
discs = 4                               # [2,6]

# the gambler

win_probability = 0.5                   # [0, 1.0]