from encoder import Encoder
from board import Board
from move import Move
from gamestate import GameState
import numpy as np

class ExtendedEncoder(Encoder):

    def __init__(self,size=9):
        self.size=size
    
    def encode_board(self,gamestate):
        np_black= np.zeros(((self.size),(self.size)),dtype=np.float32)
        np_white = np.zeros(((self.size),(self.size)),dtype=np.float32)
        np_one = np.zeros(((self.size),(self.size)),dtype=np.float32)
        np_two = np.zeros(((self.size),(self.size)),dtype=np.float32)
        np_three = np.zeros(((self.size),(self.size)),dtype=np.float32)
        for i in range(1,self.size+1):
            for j in range(1,self.size+1):
                point= (i,j)
                if point in gamestate.board.occupied:
                    group = gamestate.board.grid[(point)]
                    if group.color == 1:
                        np_black[i-1][j-1] = 1
                    else:
                        np_white[i-1][j-1] = 1
                    liberties = group.liberties
                    if liberties==1:
                        np_one[i-1][j-1] = 1
                    elif liberties ==2:
                        np_two[i-1][j-1] = 1
                    elif liberties == 3:
                        np_three[i-1][j-1] = 1

        if gamestate.turn == 1:
            board_array = np.array([np_black, np_white, np_one, np_two, np_three])
        else:
            board_array = np.array([np_white, np_black, np_one, np_two, np_three])
        return board_array

    def decode_board(self,board):
        raise NotImplementedError
        
    def encode_move(self,move):
        raise NotImplementedError

    def decode_move(self,np_move):
        raise NotImplementedError

