'''
board encoded representing the game state by 5 channels
'''
from encoders.encoder import Encoder
from gameplay.board import Board
from gameplay.move import Move
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
                    liberties = group.number_of_liberties()
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

    def decode_board(self,np_array):
        np_black = np_array[0]
        np_white = np_array[1]
        size = np_black.shape[0]

        board = Board(9)
        for i in range(size):
            for j in range(size):
                if np_black[i][j] == 1:
                    board.move(Move(1,i+1,j+1))
                elif np_white[i][j] == 1:
                    board.move(Move(-1,i+1,j+1))
        return board
        
    def encode_move(self,move):
        np_move = (move.y-1)*self.size+move.x-1
        return np_move

    def decode_move(self,np_move,turn):
        mv_x = (np_move % self.size)+1
        mv_y = int(np_move/self.size)+1
        return Move(turn,mv_x,mv_y)

