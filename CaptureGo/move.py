class Move:
    def __init__(self,turn,x,y):
        self.x = x
        self.y = y
        self.turn = turn
    def print(self):
        print("Turn:",self.turn,"X coord:",self.x,"Y coord",self.y)

