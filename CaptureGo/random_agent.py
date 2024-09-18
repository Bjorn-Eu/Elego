from agent import Agent
from gamestate import GameState
from board import Board
from move import Move
from random import randint

class RandomAgent(Agent):
    def __init__(self):
        pass

    def select_move(self,gamestate):
        legal_moves = gamestate.legal_moves()
        number_of_moves = len(legal_moves)
        index = randint(0,number_of_moves-1)
        return legal_moves[index]
