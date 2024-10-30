from gameplay.board import Board
from gameplay.gamestate import GameState
from gameplay.move import Move
import random

def atari_go9x9():
    board = Board(9)
    board.move((Move(1,4,4)))
    board.move(Move(1,5,5))
    board.move(Move(-1,5,4))
    board.move(Move(-1,4,5))
    return board

def atari_go9x9r():
    board = Board(9)
    board.move((Move(-1,4,4)))
    board.move(Move(-1,5,5))
    board.move(Move(1,5,4))
    board.move(Move(1,4,5))
    return board



def atari_go9x9B():
    board = Board(9)
    board.move((Move(1,5,4)))
    board.move(Move(1,6,5))
    board.move(Move(-1,6,4))
    board.move(Move(-1,5,5))
    return board

def atari_go9x9Br():
    board = Board(9)
    board.move((Move(-1,5,4)))
    board.move(Move(-1,6,5))
    board.move(Move(1,6,4))
    board.move(Move(1,5,5))
    return board

def atari_go9x9C():
    board = Board(9)
    board.move((Move(1,4,5)))
    board.move(Move(1,5,6))
    board.move(Move(-1,5,5))
    board.move(Move(-1,4,6))
    return board

def atari_go9x9Cr():
    board = Board(9)
    board.move((Move(-1,4,5)))
    board.move(Move(-1,5,6))
    board.move(Move(1,5,5))
    board.move(Move(1,4,6))
    return board

def atari_go9x9D():
    board = Board(9)
    board.move((Move(1,5,5)))
    board.move(Move(1,6,6))
    board.move(Move(-1,6,5))
    board.move(Move(-1,5,6))
    return board

def atari_go9x9Dr():
    board = Board(9)
    board.move((Move(-11,5,5)))
    board.move(Move(-1,6,6))
    board.move(Move(1,6,5))
    board.move(Move(1,5,6))
    return board

def capture():
    board = Board(9)
    board.move(Move(1,5,4))
    board.move(Move(1,4,5))
    board.move(Move(1,6,5))
    board.move(Move(-1,5,5))
    return board

def empty_board9x9():
    return Board(9)

def random_board9x9():
    r_var = random.random()
    board = None
    if r_var < 1/9:
        board = atari_go9x9()
    elif r_var < 2/9:
        board = atari_go9x9B()
    elif r_var < 3/9:
        board = atari_go9x9C()
    elif r_var < 4/9:
        board = atari_go9x9D()
    elif r_var < 5/9:
        board = atari_go9x9r()
    elif r_var < 6/9:
        board = atari_go9x9Br()
    elif r_var < 7/9:
        board = atari_go9x9Cr()
    elif r_var < 8/9:
        board = atari_go9x9Dr()
    else:
        board = empty_board9x9()
    return board


