from agent import Agent
from board import Board
from gamestate import GameState
from move import Move
class HumanAgent(Agent):

    def select_move(self,gamestate):
        while True:
            gamestate.print_state()
            move_string = input("Enter move:").lower() 
        
            if len(move_string) == 2:
                move_xy = list(move_string)
                move_x = ord(move_xy[0])-ord('a')+1
                move_y = ord(move_string[1])-ord('0')
                move = Move(gamestate.turn,move_x,move_y)
                if gamestate.is_legal_move(move):
                    return move
