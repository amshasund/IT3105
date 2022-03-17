from anet import ANet
from hex import Hex
from random import seed
seed(1)

def main():
    # TESTS
    #anet_test = ANet()
    hex_test = Hex()
    hex_test.init_game_board()
    # Player 2
    hex_test.perform_move([hex_test.current_player, (0, 2)])
    # Player 1
    hex_test.perform_move([hex_test.current_player, (1, 0)])
    # Player 2
    hex_test.perform_move([hex_test.current_player, (0, 1)])
    # Player 1
    hex_test.perform_move([hex_test.current_player, (1, 1)])
    # Player 2
    hex_test.perform_move([hex_test.current_player, (2, 0)])
    # Palyer 1
    hex_test.perform_move([hex_test.current_player, (1, 2)])
    # Player 2

    # search
    is_game_over = hex_test.game_over()
    print(is_game_over)

    hex_test.print_game_board()

main()