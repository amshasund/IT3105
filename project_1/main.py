from rl_system import RLSystem


def main():
    """Runs the game"""
    print("Let's play!")
    # Create a RL system
    rl_system = RLSystem()

    # The Game Loop
    rl_system.actor_critic_algorithm()

    # Print the result
    rl_system.sim_world.print_end_results(rl_system.actor.policy)


main()
