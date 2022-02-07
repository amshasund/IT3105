from rl_system import RLSystem


def main():
    """Runs the game"""
    print("Let's play!")
    # Create A RL system
    rl_system = RLSystem()

    print(rl_system.critic.state_to_binary(93))
    print(rl_system.critic.state_to_binary([-3, -54, 103, 4]))
    print(rl_system.critic.state_to_binary([3, 54, 103, 4]))
    print(rl_system.critic.binary_to_state("01011101"))
    print(rl_system.critic.binary_to_state("11111101110010100110011100000100"))
    print(rl_system.critic.binary_to_state("00000011001101100110011100000100"))

    # The Game Loop
    # rl_system.actor_critic_algorithm()

    # Print the result
    # rl_system.sim_world.print_results(rl_system.actor.policy)


main()
