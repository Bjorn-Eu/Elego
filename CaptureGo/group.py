

class Group():
    def __init__(self,color):
        self.color = color
        self.stones = set()
        self.liberties = set()

    def contains_stone(self,stone):
        return self.stones.get(stone)

    def merge_group(self,group):
        self.stones.update(group.stones)
        self.liberties.update(group.liberties)

    def add_stone(self,stone):
        self.stones.add(stone)

    def add_liberties(self,liberties):
        self.liberties.update(liberties)


    def number_of_liberties(self):
        return len(self.liberties)
