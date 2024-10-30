from gameplay.board import Board
from gameplay.move import Move
import copy

class GameState:
    def __init__(self,turn,board):
        self.turn = turn
        self.board = board
        self.move_history = []
        self.was_capture = False
        self.winner = 0
    
    def print_state(self):
        print("It is player", turn_to_string(self.turn), "turn to play")
        self.board.print_board()

    def legal_moves(self):
        if self.was_capture:
            return []
        else:
            return self.board.legal_moves(self.turn)

    def is_legal_move(self,move):
        if move.turn == self.turn:
            return (self.board).is_legal_move(move)
        else:
            print("Not your turn to play")
            return False

    #returns the score of the position
    def score(self):
        pass

    def is_over(self):
        legal_moves = self.legal_moves()
        out_of_moves = (len(legal_moves) == 0)
        return (self.was_capture or out_of_moves)

    def move(self,move):
        if self.board.is_capture(move):
            self.was_capture = True
            self.winner = -self.turn
            #self.turn = 0

        self.board.move(move)
        self.move_history.append(move)
        self.switch_turn()
        
    def switch_turn(self):
        if self.turn == Board.BLACK:
            self.turn = Board.WHITE
        elif self.turn == Board.WHITE:
            self.turn = Board.BLACK
    
    def reverse_game_state(self):
        reversed_board = self.board.reverse_board()
        reversed_game_state = GameState(-gamestate.turn,reversed_board)
        #add reversal of move history etc.

    #undos a non-capturing move
    def undo(self):
        #if len(self.move_history) != 0:
            last_move = self.move_history.pop() 
            self.board.move(Move(0,last_move.x,last_move.y))
            self.switch_turn()
            self.was_capture = False

    

    def evaluate(self):
        if self.was_capture == True:
            return -self.turn
        else:
            return 0
    
    def __deepcopy__(self,memo={}):
        new_board = copy.deepcopy(self.board)
        return GameState(self.turn,new_board)

def turn_to_string(turn):
        if turn == 1:
            return "Black"
        elif turn == -1:
            return "White"
        else:
            return "Game ended"
    






        