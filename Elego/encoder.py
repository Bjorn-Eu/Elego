from board import Board
from move import Move

class Encoder():
    def encode_board(self,board):
        raise NotImplementedError

    def decode_board(self,np_board):
        raise NotImplementedError
    
    def encode_move(self,move):
        raise NotImplementedError

    def decode_move(self,np_move):
        raise NotImplementedError
