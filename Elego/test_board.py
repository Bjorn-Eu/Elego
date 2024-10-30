import gameplay.board
from gameplay.board import Board
from gameplay.move import Move



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
    
   
    board.print_board()

    group = board.grid[(3,2)]
    assert(len(group.stones) == 6)
    board.move(Move(1,3,3))
    board.print_board()
    
    group = board.grid[(3,2)]
    assert(len(group.stones) == 13)

def test_connected2():
    board = Board(9)
    for i in range (1,10):
        for j in range(1,4):
            board.move(Move(1,i,j))

    group = board.grid[(3,2)]
    board.print_board()
    assert(len(group.stones) == 27)
    liberties = group.number_of_liberties()
    assert(liberties == 9)

    

def test_connected3():
    board = Board(5)
    for i in range(1,6):
        board.move(Move(-1,i,1))
        board.move(Move(-1,1,i))
        board.move(Move(-1,i,5))
        board.move(Move(-1,5,i))

    board.print_board()
    group = board.grid[(2,1)]
    assert(len(group.stones) == 16)
    assert(group.number_of_liberties()== 8)


def test_connected4():
    board = Board(5)
    for i in range(2,5):
        board.move(Move(-1,i,2))
        
        board.move(Move(-1,i,4))
        
    board.move(Move(-1,2,3))
    board.move(Move(-1,4,3))
    board.print_board()
    group = board.grid[(3,2)]

    assert(len(group.stones) == 8)
    print(group)
    assert(group.number_of_liberties() == 13)
    

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
    group = board.grid[(3,2)]
    assert(len(group.stones)==5)
    assert(group.number_of_liberties() == 7)

    group = board.grid[(4,1)]
    assert(len(group.stones)==2)
    assert(group.number_of_liberties() == 1)


    move = Move(-1,3,1)
    assert(board.is_capture(move))
    move = Move(1,3,1)


    assert(board.is_self_capture(move))
    assert(not board.is_legal_move(move))

  
def test_bigcapture():
    board = Board(9)
    for i in range (1,10):
        for j in range(1,9):
            board.move(Move(1,i,j))
            

    for i in range (1,9):
        board.move(Move(-1,i,9))
    board.print_board()
    assert(board.is_capture(Move(-1,9,9)))
    assert(board.is_legal_move(Move(-1,9,9)))
    assert(board.is_capture(Move(1,9,9)))
    assert(board.is_legal_move(Move(1,9,9)))


def test_bigcapture2():
    board = Board(9)
    for i in range (1,10):
        for j in range(2,9):
            board.move(Move(1,i,j))
    for i in range (1,9):
        board.move(Move(-1,i,9))
        board.move(Move(1,i+1,1))
    assert(not board.is_capture(Move(-1,9,9)))
    assert(not board.is_legal_move(Move(-1,9,9)))
    assert(board.is_capture(Move(1,9,9)))
    assert(board.is_legal_move(Move(1,9,9)))
