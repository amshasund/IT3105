from rl_system import RLSystem


def main():
    """Runs the game"""
    print("Let's play!")
    # Create a RL system
    rl_system = RLSystem()

    rl_system.sim_world.environment.print_game_board()
    rl_system.sim_world.environment.get_options()
    rl_system.sim_world.environment.update_game_board(1, 2)
    rl_system.sim_world.environment.print_game_board()
    rl_system.sim_world.environment.get_options()

    # The Game Loop
    rl_system.actor_critic_algorithm()

    # Print the result
    rl_system.sim_world.print_end_results(rl_system.actor.policy)


main()
