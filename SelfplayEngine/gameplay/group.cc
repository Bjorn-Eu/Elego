#include "group.h"


void Group::remove_liberty(const Point liberty){
    liberties.erase(liberty);
}

void Group::merge_group(const Group group){
    stones.insert(group.stones.begin(), group.stones.end());
    liberties.insert(group.liberties.begin(), group.liberties.end());
}

int Group::number_of_liberties() const{
    return liberties.size();
}


int Group::number_of_stones() const{
    return stones.size();
}

