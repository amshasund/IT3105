from anet import ANet
from hex import Hex

def main():
    # TESTS
    #anet_test = ANet()
    hex_test = Hex()
    hex_test.init_game_board()
    hex_test.print_game_board()
    print(hex_test.current_player)
    hex_test.perform_move([hex_test.current_player, (1, 0)])
    hex_test.print_game_board()
    print(hex_test.current_player)
    hex_test.perform_move([hex_test.current_player, (2, 1)])
    hex_test.print_game_board()
    print(hex_test.current_player)
    
main()