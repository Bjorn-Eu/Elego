import board
from board import Board
from move import Move
import collections



def test_board1():
    board = Board(5)
    move = Move(1,1,1)
    board.move(Move(1,1,1))
    assert(not board.is_legal_move(move))
    move = Move(-1,1,1)
    assert(not board.is_legal_move(move))

    legal_moves = board.legal_moves(1)
    assert(len(legal_moves) == 24)

#test capturing
def test_board():
    board = Board(5)
    board.move(Move(-1,1,2))
    board.move(Move(1,1,1))
    move = Move(-1,2,1)
    assert(board.is_capture(move))

def test_connected():
    board = Board(5)
    for i in range(1,6):
        board.move(Move(1,i,1))
        board.move(Move(1,i,5))
    board.move(Move(1,3,2))
    board.move(Move(1,3,4))
    
    assert(len(board.adjacent_friends(1,1,1)) == 1)
    assert(len(board.adjacent_friends(1,3,1)) == 3)
    board.print_board()
    group1 = board.connected_friends(1,[3, 1],collections.deque([[3, 1]]))
    group2 = board.connected_friends(1,[3, 5],collections.deque([[3, 5]]))
    assert(len(group1) == 6)
    assert(len(group2) == 6)
    board.move(Move(1,3,3))
    board.print_board()
    group = board.connected_friends(1,[3, 1],collections.deque([[3, 1]]))
    assert(len(group) == 13)

def test_connected2():
    board = Board(9)
    for i in range (1,10):
        for j in range(1,4):
            board.move(Move(1,i,j))

    group = board.connected_friends(1,[1,1],collections.deque([[1, 1]]))
    board.print_board()
    print(group)
    assert(len(group) == 27)
    liberties = board.liberties(1,1)
    lib2 = board.total_liberties(group)
    assert(liberties == 0)
    assert(lib2 == 9)

    board.move(Move(0,5,3))
    group = board.connected_friends(1,[1,1],collections.deque([[1, 1]]))
    board.print_board()
    print(group)
    assert(len(group) == 26)
    liberties = board.liberties(1,1)
    lib2 = board.total_liberties(group)
    assert(liberties == 0)
    assert(lib2 == 9)

def test_connected3():
    board = Board(5)
    for i in range(1,6):
        board.move(Move(-1,i,1))
        board.move(Move(-1,1,i))
        board.move(Move(-1,i,5))
        board.move(Move(-1,5,i))
    group = board.connected_friends(-1,[1,3],collections.deque([[1,3]]))
    assert(len(group) == 16)
    assert(board.total_liberties(group) == 8)


def test_connected4():
    board = Board(5)
    for i in range(2,5):
        board.move(Move(-1,i,2))
        board.move(Move(-1,2,i))
        board.move(Move(-1,i,4))
        board.move(Move(-1,4,i))
    group = board.connected_friends(-1,[2,3],collections.deque([[2,3]]))
    assert(len(group) == 8)
    print(group)
    assert(board.total_liberties(group) == 13)
    
    
    
    

def test1():
    board = Board(5)
    board.move(Move(-1,1,2))
    board.move(Move(-1,2,1))
    board.move(Move(1,1,3))
    board.move(Move(1,2,2))
    board.move(Move(1,3,1))
    board.print_board()
    move = Move(1,1,1)
    assert(board.is_capture(move))
    assert(board.is_legal_move(move))






#test ponnuki
def test_boarda():
    board = Board(5)
    board.move(Move(-1,1,3))
    board.move(Move(-1,2,2))
    board.move(Move(-1,2,4))

    board.move(Move(-1,3,3))


    board.print_board()
    move = Move(1,2,3)
    assert(not board.is_legal_move(move))
    assert(board.is_self_capture(move))
    move = Move(-1,2,3)
    assert(board.is_legal_move(move))
    assert(board.is_self_capture(move)==False)
    assert(not board.is_capture(move))

#turtleshape
def test_boardb():
    board = Board(5)
    board.move(Move(-1,1,3))
    board.move(Move(-1,2,2))
    board.move(Move(-1,3,2))
    board.move(Move(-1,2,4))
    board.move(Move(-1,3,4))
    board.move(Move(-1,4,3))
    
    board.move(Move(1,2,3))
    board.print_board()
    move = Move(1,3,3)
    assert(not board.is_legal_move(move))

#test self capturing
def test_board3a():
    board = Board(5)
    board.move(Move(-1,1,2))
    board.move(Move(-1,2,1))
    move = Move(1,1,1)
    assert(board.is_self_capture(move))

def test_board3b():
    board = Board(5)
    board.move(Move(1,1,2))
    board.move(Move(1,2,1))
    move = Move(-1,1,1)
    assert(board.is_self_capture(move))


def test_board2():
    board = Board(5)
    board.move(Move(-1,2,1))
    board.move(Move(-1,2,2))
    board.move(Move(-1,3,2))
    board.move(Move(-1,4,2))
    board.move(Move(-1,5,2))
    board.move(Move(1,4,1))
    board.move(Move(1,5,1))
    board.print_board()
    groupw = board.connected_friends(-1,[2, 1],collections.deque([[2, 1]]))
    assert(not board.has_zero_liberties(groupw))
    assert(len(groupw)==5)
    assert(board.total_liberties(groupw) == 7)

    groupb = board.connected_friends(1,[5, 1],collections.deque([[5, 1]]))
    assert(not board.has_zero_liberties(groupb))
    assert(len(groupb)==2)
    assert(board.total_liberties(groupb) == 1)


    move = Move(-1,3,1)
    assert(board.is_capture(move))
    move = Move(1,3,1)

    assert(board.is_self_capture(move))
    #assert(not board.is_legal_move(move))

    board.move(move)
    groupb = board.connected_friends(1,[5, 1],collections.deque([[5, 1]]))
    assert(len(groupb) == 3)
    assert(board.total_liberties(groupb) == 0)
    assert(board.has_zero_liberties(groupb))    
  
def test_bigcapture():
    board = Board(9)
    for i in range (1,10):
        for j in range(1,9):
            board.move(Move(1,i,j))
            

    for i in range (1,9):
        board.move(Move(-1,i,9))
    
    assert(board.is_capture(Move(-1,9,9)))
    assert(board.is_legal_move(Move(-1,9,9)))
    assert(board.is_capture(Move(1,9,9)))
    assert(board.is_legal_move(Move(1,9,9)))

    board.move(Move(0,1,1))
    assert(not board.is_capture(Move(-1,9,9)))
    assert(not board.is_legal_move(Move(-1,9,9)))
    assert(board.is_capture(Move(1,9,9)))
    assert(board.is_legal_move(Move(1,9,9)))

