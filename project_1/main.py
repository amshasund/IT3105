from rl_system import RLSystem


def main():
    """Runs the game"""
    print("Let's play!")
    # Create A RL system
    rl_system = RLSystem()
    rl_system.sim_world.environment.print_game_board()

    # The Game Loop
    # rl_system.actor_critic_algorithm()

    # Print the result
    # rl_system.sim_world.print_results(rl_system.actor.policy)


main()
