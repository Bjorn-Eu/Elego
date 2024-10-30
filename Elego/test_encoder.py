'''
import gameplay.board
from gameplay.board import Board
from gameplay.gamestate import GameState
from gameplay.move import Move
from encoders.extendedencoder import ExtendedEncoder
import gameplay.config_board as config_board

def test_lib0():
    board = config_board.atari_go9x9()
    gamestate = GameState(1,board)
    encoder = ExtendedEncoder()
    encoded_board = encoder.encode_board(gamestate)
    gamestate.print_state()
    print(encoded_board[3][4])
    assert(encoded_board[3][4][3] == 1)   

'''
