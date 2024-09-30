from encoder import Encoder
from board import Board
from move import Move
from gamestate import GameState
import numpy as np

class SimpleEncoder(Encoder):
    

    def __init__(self,size=9):
        self.size=size
    
    def encode_board(self,gamestate):
        np_black= np.zeros(((self.size),(self.size)),dtype=np.float32)
        np_white = np.zeros(((self.size),(self.size)),dtype=np.float32)
        for i in range(1,self.size+1):
            for j in range(1,self.size+1):
                point= (i,j)
                if point in gamestate.board.occupied:
                    if gamestate.board.grid[(point)].color == 1:
                        np_black[i-1][j-1] = 1
                    else:
                        np_white[i-1][j-1] = 1

        if gamestate.turn == 1:
            board_array = np.array([np_black, np_white])
        else:
            board_array = np.array([np_white, np_black])
        return board_array

    def decode_board(self,board):
        raise NotImplementedError
        
    def encode_move(self,move):
        raise NotImplementedError

    def decode_move(self,np_move):
        raise NotImplementedError



