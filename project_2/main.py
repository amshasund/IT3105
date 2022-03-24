from anet import ANet
from hex import Hex
from random import seed
seed(1)


def main():
    # TESTS
    #anet_test = ANet()
    hex_test = Hex()
    hex_test.init_game_board()
    # Player 1
    hex_test.perform_move([hex_test.current_player, (0, 0)])
    # Player 2
    hex_test.perform_move([hex_test.current_player, (0, 1)])
    # Player 1
    hex_test.perform_move([hex_test.current_player, (0, 2)])
    # Player 2
    hex_test.perform_move([hex_test.current_player, (1, 1)])
    # Player 1
    hex_test.perform_move([hex_test.current_player, (1, 0)])
    # Player 2
    hex_test.perform_move([hex_test.current_player, (2, 0)])

    # Print game board
    hex_test.print_game_board(hex_test.get_hex_board())


    # search
    is_game_over = hex_test.game_over()
    if is_game_over:
        print("Someone has won.")
    else:
        print("Game is not over")


main()
