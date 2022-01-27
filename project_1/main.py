from rl_system import RLSystem
import random
from parameters import win_probability


def main():
    """ Runs the game """
    print("Let's play!")
    # Create A RL system
    rl_system = RLSystem()

    # The Game Loop
    rl_system.actor_critic_algorithm()

    # Print the result
    rl_system.sim_world.print_results(rl_system.actor.policy)


main()
