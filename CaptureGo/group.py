
import copy
class Group():
    def __init__(self,color,stones,liberties):
        self.color = color
        self.stones = frozenset(stones)
        self.liberties = frozenset(liberties)

    def contains_stone(self,stone):
        return self.stones.get(stone)

    def merge_group(self,group):
        stones = self.stones | group.stones
        liberties = self.liberties | group.liberties
        return Group(self.color,stones,liberties)

    def add_stone(self,stone):
        self.stones.add(stone)

        self.liberties.update(liberties)

    def remove_liberty(self,liberty):
        new_liberties = self.liberties.difference(frozenset([liberty]))
        return Group(self.color,self.stones,new_liberties)


    def number_of_liberties(self):
        return len(self.liberties)

    def __deepcopy__(self,memo={}):
        return Group(self.color,self.stones,self.liberties)
