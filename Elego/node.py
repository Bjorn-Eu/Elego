import numpy as np
from move import Move
class Node():
    def __init__(self,gamestate,parent,last_move,priors,value):
        self.gamestate = gamestate
        self.parent = parent
        self.last_move = last_move
        self.visits = 1
        self.value = value
        self.size = gamestate.board.size
        self.is_terminal = gamestate.is_over()
        self.virtual_losses = 0

        self.branch_values = np.zeros((self.size*self.size),dtype=np.float32)
        self.branch_priors = priors
        self.branch_visits = np.zeros((self.size*self.size),dtype=np.float32)

        self.children = {}
        self.branches = set()

        #create set of legal branches
        for mv_index in range(self.size*self.size):    
            mv_x = mv_index%self.size + 1
            mv_y = mv_index//self.size+1
            move = Move(gamestate.turn,mv_x,mv_y)
            if(gamestate.is_legal_move(move)):
                self.branches.add(mv_index)

    def get_total_value(self):
        return self.values

    def set_total_value(self,values):
        self.values = values

    def get_visit_count(self):
        return self.branch_visits

    def set_visit_count(self,visits):
        self.branch_visits = visits

    def select_branch(self):
        moves = np.argsort(self.score_branch())[::-1] #sort by descending order
        for move in moves:
            if move in self.branches:
                return move

    def select_move(self):
        return np.argmax(self.branch_visits)

    def add_child(self,move,child):
        self.children[move] = child
        #self.visits += 1
        #self.branch_visits[move] += 1
        #self.branch_values[move] -= child.value

    def has_child(self,move):
        return move in self.children

    def get_child(self,move):
        return self.children[move]
            
    def score_branch(self):
        return self.branch_Q() + self.branch_U()

    def branch_Q(self):
        return np.divide(self.branch_values, self.branch_visits,
        out=np.zeros_like(self.branch_values), where=self.branch_visits!=0)

    def branch_U(self):
        c=1.4
        return c*self.branch_priors*np.sqrt(self.visits)/(self.branch_visits+1)

    def update_wins(self,value):
        if not self.parent is None:
            parent = self.parent
            parent.visits = parent.visits + 1
            parent.branch_visits[self.last_move] += 1
            parent.branch_values[self.last_move] += value
            parent.update_wins(-value)

    def add_virtual_loss(self):
        self.virtual_losses += 1
        self.visits += 1
        self.value -= 1
        if not self.parent is None:
            self.parent.branch_values[self.last_move] -= 1
            self.parent.branch_visits[self.last_move] += 1
            self.parent.add_virtual_loss()

    def undo_virtual_losses(self):
        self.visits -= self.virtual_losses
        self.value += self.virtual_losses
        if not self.parent is None:
            self.parent.branch_values[self.last_move] += self.virtual_losses
            self.parent.branch_visits[self.last_move] -= self.virtual_losses
            self.parent.undo_virtual_losses()
        self.virtual_losses = 0

    def print(self):
        self.gamestate.print_state()
        print("Number of visits:",self.visits)
        print("Value:",self.value)
        print("Branch visits:",self.branch_visits)
        print("Branch values:",self.branch_values)
        print("Branch priors:",self.branch_priors)
        if not self.branch_priors is None:
            print("Explore score:",(self.score_branch()))