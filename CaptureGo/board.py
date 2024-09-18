from move import Move
import random
import numpy as np

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


    def move(self,move):
        self.board[move.x][move.y] = move.turn

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
        if self.board[index_x+1][index_y] == 0:
            liberties += 1
        if self.board[index_x-1][index_y] == 0:
            liberties += 1
        if self.board[index_x][index_y+1] == 0:
            liberties += 1
        if self.board[index_x][index_y-1] == 0:
            liberties +=1
        return liberties

    #assuming move isn't a capture it checks if it is a self capture
    def is_self_capture(self,move):
        self.move(move)
        group = []
        group.append([move.x,move.y])
        group = self.connected_friends(move.turn,[move.x,move.y],group)
        self_capture = self.has_zero_liberties(group)
        mv = Move(0,move.x,move.y)
        self.move(mv)
        return self_capture
    
    def is_capture(self,move):
        enemies = []
        ENEMY = 0
        if move.turn ==self.BLACK:
            enemies = self.adjacent_friends(self.WHITE,move.x,move.y)
            ENEMY = self.WHITE
        else:
            enemies = self.adjacent_friends(self.BLACK,move.x,move.y)
            ENEMY = self.BLACK
 
        for enemy in enemies:
            group = []
            group.append(enemy)
            group = self.connected_friends(ENEMY,enemy,group)
            if self.total_liberties(group) == 1:
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
    
    #note this double counts liberties but as it only checks for zero it works
    def has_zero_liberties(self,group):
        liberties = 0
        for friend in group:
            liberties += self.liberties(friend[0],friend[1])
        if liberties==0:
            return True
        else:
            return False

    #returns deque of all coordinate of adjacent stone of that color
    def adjacent_friends(self,color,index_x,index_y):
        friends = []
        if self.board[index_x+1][index_y] == color:
            friends.append([index_x+1,index_y])
        if self.board[index_x-1][index_y] == color:
            friends.append([index_x-1,index_y])
        if self.board[index_x][index_y+1] == color:
            friends.append([index_x,index_y+1])
        if self.board[index_x][index_y-1] == color:
            friends.append([index_x,index_y-1])
        return friends

    #returns a deque of all legal moves for a given color
    def legal_moves(self,color):
        moves = []
        for i in range(1,self.size+1):
            for j in range(1,self.size+1):
                move = Move(color,i,j)
                if self.is_legal_move(move):
                    moves.append(move)
        return moves

    def reverse_board(self):
        reversed_board = Board(self.size)
        for i in range(1,self.size+1):
            for j in range(1,self.size+1):
                reversed_board.board[i][j] = -self.board[i][j]

        return reversed_board
        
