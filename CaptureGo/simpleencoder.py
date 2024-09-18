from encoder import Encoder
from board import Board
from move import Move
from gamestate import GameState
import numpy as np

class SimpleEncoder(Encoder):
    
    def __init__(self,size=9):
        self.size=size
    
    def encode_board(self,gamestate):
        np_board = np.array((gamestate.board).board)
        np_inner_board = np_board[1:(self.size+1),1:(self.size+1)]
        
        np_black = np.copy(np_inner_board)
        np_white = np.copy(np_inner_board)
        np_black[np_black==-1] = 0
        np_white[np_white==1] = 0
        np_white = -np_white

        board_array = None
        if gamestate.turn == 1:
            board_array = np.array([np_black, np_white])
        else:
            board_array = np.array([np_white, np_black])
        board_array = board_array.astype('float32')
        return board_array

    def decode_board(self,board):
        board = Board(self.size)



        raise NotImplementedError
    def encode_move(self,move):
        raise NotImplementedError

    def decode_move(self,np_move):
        raise NotImplementedError


