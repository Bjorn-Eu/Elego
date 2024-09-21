from move import Move
import random
import numpy as np
from group import Group

class Board:
    EMPTY = 0
    BLACK = 1
    WHITE = -1
    BORDER = 3
    def __init__(self,size):
        self.size=size
        self.arraysize = size+2
        #make a board of dimension size*size with a one space border around it
        #self.board = np.zeros((self.arraysize, self.arraysize),dtype=np.int32) seems this is just slower in general
        self.board = [[self.EMPTY for col in range(self.arraysize)] for row in range(self.arraysize)]

        #set all border squares
        self.board[0] = [self.BORDER for col in range(self.arraysize)]
        self.board[size+1] = self.board[0]
        for i in range(size+1):
            self.board[i][0] = self.BORDER
            self.board[i][size+1] = self.BORDER
        self.groups = []


    def move(self,move):
        self.board[move.x][move.y] = move.turn
        new_group = self.create_stone(move)
        liberties = self.liberties(move.x,move.y)
        if liberties != 4:
            for group in self.groups:
                group.liberties.discard((move.x, move.y))
        adjacents = self.adjacent_friends(move.turn,move.x,move.y)
        for adjacent in adjacents:
            for group in self.groups:       
                if adjacent in group.stones and (adjacent not in new_group.stones):
                    new_group.merge_group(group)
                    self.groups.remove(group)
        
        self.groups.append(new_group)


    def create_stone(self,move):
        group = Group(move.turn)
        group.add_stone((move.x, move.y))
        adjacent_lib = self.adjacent_liberties(move)
        group.add_liberties(adjacent_lib)
        return group

    def adjacent_liberties(self,move):
        index_x = move.x
        index_y = move.y
        liberties = set()
        if self.board[index_x+1][index_y] == self.EMPTY:
            liberties.add((index_x+1,index_y))
        if self.board[index_x-1][index_y] == self.EMPTY:
            liberties.add((index_x-1,index_y))
        if self.board[index_x][index_y+1] == self.EMPTY:
            liberties.add((index_x,index_y+1))
        if self.board[index_x][index_y-1] == self.EMPTY:
            liberties.add((index_x,index_y-1))
        return liberties


    def is_legal_move(self,move):
        if self.board[move.x][move.y] != 0:
            return False
        elif self.is_capture(move):
            return True
        elif self.is_self_capture(move):
            return False
        else:
            return True

    #returns the number of adjacent empty spaces
    def liberties(self,index_x,index_y):
        liberties = 0
        if self.board[index_x+1][index_y] == self.EMPTY:
            liberties += 1
        if self.board[index_x-1][index_y] == self.EMPTY:
            liberties += 1
        if self.board[index_x][index_y+1] == self.EMPTY:
            liberties += 1
        if self.board[index_x][index_y-1] == self.EMPTY:
            liberties +=1
        return liberties
    
    #assuming move isn't a capture it checks if it is a self capture
    def is_self_capture(self,move):
        if self.liberties(move.x,move.y) > 0:
            return False
        else:
            group = self.create_stone(move)
            adjacents = self.adjacent_friends(move.turn,move.x,move.y)
            for adjacent in adjacents:
                for group in self.groups:
                    if adjacent in group.stones:
                        if group.number_of_liberties() >= 2:
                            return False
        return True

    
    def is_capture(self,move):
        ENEMY = -move.turn
        enemies = self.adjacent_friends(ENEMY,move.x,move.y)
        
        for enemy in enemies:
            for group in self.groups:
                if (enemy in group.stones) and group.number_of_liberties()==1:
                    return True
        return False

    def print_board(self):
        for i in range(1,self.size+1):
            for j in range(1,self.size+1):
                color = self.board[j][i]
                if color == self.EMPTY:
                    print(".",end=" ")
                elif color == self.BLACK:
                    print("B",end=" ")
                else:
                    print("W",end=" ")    
            print("")

    #counts the total liberties of a group of stones
    def total_liberties(self,group):
        temp = []
        liberties = 0
        for x in group:
            adjacent = self.adjacent_friends(0,x[0],x[1])
            for i in adjacent:
                if not i in temp:
                    temp.append(i)
                    liberties += 1
        return liberties

    def connected_friends(self,turn,coord,friends):
        temp_friends = self.adjacent_friends(turn,coord[0],coord[1])
        for x in temp_friends:
            if not x in friends:
                friends.append(x)
                self.connected_friends(turn,x,friends)
        return friends

    def adjacent_stones(self,index_x,index_y):
        friends = []
        if self.board[index_x+1][index_y] != 0:
            friends.append((index_x+1,index_y))
        if self.board[index_x-1][index_y] != 0:
            friends.append((index_x-1,index_y))
        if self.board[index_x][index_y+1] != 0:
            friends.append((index_x,index_y+1))
        if self.board[index_x][index_y-1] != 0:
            friends.append((index_x,index_y-1))
        return friends

    #returns list of all coordinate of adjacent stone of that color
    def adjacent_friends(self,color,index_x,index_y):
        friends = []
        if self.board[index_x+1][index_y] == color:
            friends.append((index_x+1,index_y))
        if self.board[index_x-1][index_y] == color:
            friends.append((index_x-1,index_y))
        if self.board[index_x][index_y+1] == color:
            friends.append((index_x,index_y+1))
        if self.board[index_x][index_y-1] == color:
            friends.append((index_x,index_y-1))
        return friends

    #returns a list of all legal moves for a given color
    def legal_moves(self,color):
        moves = []
        for i in range(1,self.size+1):
            for j in range(1,self.size+1):
                move = Move(color,i,j)
                if self.is_legal_move(move):
                    moves.append(move)
        return moves

        
