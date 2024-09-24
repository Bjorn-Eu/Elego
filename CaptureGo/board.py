from move import Move
import random
import numpy as np
from group import Group
import copy


class Board:
    EMPTY = 0
    BLACK = 1
    WHITE = -1
    BORDER = 3
    def __init__(self,size,adjacency_table = None):
        self.size=size
        self.occupied = set()
        self.grid = {}
        self.adjacency_table = adjacency_table
        if self.adjacency_table is None:
            self.adjacency_table = create_adjacency_table(self.size)



    def move(self,move):
        point = (move.x, move.y)
        self.occupied.add(point)
        
        new_group = self.create_stone(move)
        adjacent_groups = self.adjacent_groups(point)
        for group in adjacent_groups: 
            for stone in group.stones:
                self.grid[stone] = self.grid[stone].remove_liberty(point)
    

        adjacent_groups = self.adjacent_groups(point)
        new_group = self.create_stone(move)
        for group in adjacent_groups:
            if group.color == move.turn:
                new_group = new_group.merge_group(group)

        for stone in new_group.stones:
            self.grid[stone] = new_group
        

    def create_stone(self,move):
        adjacent_lib = self.adjacent_liberties(move)
        group = Group(move.turn,[(move.x, move.y)],adjacent_lib)
        return group

    def adjacent_liberties(self,move):
        return self.adjacency_table[(move.x,move.y)]-self.occupied

    def adjacent_groups(self,point):
        adjacent_groups = set()
        adjacent_stones = self.adjacent_stones(point)
        for stone in adjacent_stones:
                adjacent_groups.add(self.grid[stone])
        return adjacent_groups
    
    def adjacent_stones(self,point):
        return self.adjacency_table[point] & self.occupied

    def adjacent_groups_index(self,point):
        adjacent_group_index = []
        for i in range(len(self.groups)):
            if point in self.groups[i].liberties:
                if not i in adjacent_group_index:   
                    adjacent_group_index.append(i)
        return adjacent_group_index

    def is_legal_move(self,move):
        if (move.x, move.y) in self.occupied:
            return False
        elif self.is_capture(move):
            return True
        elif self.is_self_capture(move):
            return False
        else:
            return True
    
    #assuming move isn't a capture it checks if it is a self capture
    def is_self_capture(self,move):
        group = self.create_stone(move)
        
        if group.number_of_liberties() > 0:
            return False
        else:
            adjacent_groups = self.adjacent_groups((move.x, move.y))
            for group in adjacent_groups:
                if group.color == move.turn:
                    if group.number_of_liberties() >= 2:
                        return False
        return True

    
    def is_capture(self,move):
        ENEMY = -move.turn
        adjacent_groups = self.adjacent_groups((move.x, move.y))
        for group in adjacent_groups:
            if group.color == ENEMY:
                if group.number_of_liberties() == 1:
                    return True
        return False

    def print_board(self):
        np_board= np.zeros(((self.size),(self.size)),dtype=np.float32)
        for i in range(1,self.size+1):
            for j in range(1,self.size+1):
                point= (i,j)
                if point in self.occupied:
                    np_board[i-1][j-1] = self.grid[(point)].color
        

        for i in range(0,self.size):
            for j in range(0,self.size):
                color = np_board[j][i]
                if color == self.EMPTY:
                    print(".",end=" ")
                elif color == self.BLACK:
                    print("B",end=" ")
                else:
                    print("W",end=" ")    
            print("")
        


    #returns a list of all legal moves for a given color
    def legal_moves(self,color):
        moves = []
        candidates = self.adjacency_table.keys() - self.occupied
        for candidate in candidates:
            move = Move(color,candidate[0],candidate[1])
            if self.is_legal_move(move):
                moves.append(move)
        return moves

        
    def __deepcopy__(self,memo={}):
        board = Board(self.size,adjacency_table=self.adjacency_table)
        board.occupied = copy.copy(self.occupied)
        board.grid = copy.copy(self.grid) #shallow copy
        return board
        

def create_adjacency_table(size):
    adjacency_table = {}
    
    #inside points
    for i in range(2,size):
        for j in range(2,size):
            point = (i, j)
            adjacency_table[point] = {(i-1,j),(i+1,j),(i,j-1),(i,j+1)}
    #edge points
    for i in range (2,size):
        adjacency_table[(1,i)] = {(2,i),(1,i-1),(1,i+1)}
        adjacency_table[(size,i)] = {(size-1,i),(size,i-1),(size,i+1)}
        adjacency_table[(i,1)] = {(i,2),(i-1,1),(i+1,1)}
        adjacency_table[(i,size)] = {(i,size-1),(i-1,size),(i+1,size)}
    #corner points
    adjacency_table[(1,1)] = {(1,2),(2,1)}
    adjacency_table[(1,size)] = {(2,size),(1,size-1)}
    adjacency_table[(size,size)] = {(size,size-1),(size-1,size)}
    adjacency_table[(size,1)] = {(size,2),(size-1,1)}
    return adjacency_table