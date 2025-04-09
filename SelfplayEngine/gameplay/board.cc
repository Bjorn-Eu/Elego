#include "board.h"
#include <stdio.h>




bool Board::is_legal(Move move){
    if (groups.count({move.x, move.y})){
        return false;
    }
    else if(is_capture(move)){
        return true;
    }
    else if(is_self_capture(move)){
        return false;
    }
    else{
        return true;
    }
}

void Board::play_move(Move move){
    Point point({move.x,move.y});
    std::unordered_set<Point> s = {point};
    std::unordered_set<Point> l = adjacency_table->at(point);

    std::shared_ptr<Group> group = std::make_shared<Group>(move.turn,s,l);
    
    for (auto neighbor : l){
        if (groups.count(neighbor)){
            group->remove_liberty(neighbor);
            std::shared_ptr<Group> neighbor_group = groups.at(neighbor);
            neighbor_group->remove_liberty(point);
            groups.at(neighbor)->remove_liberty(point);
            //merge adjacent groups of same color
            if(neighbor_group->color == move.turn){
                group->merge_group(*neighbor_group);
                groups.at(neighbor) = group;
            }
        }
    }
    //update the group of the point
    groups.insert({point,group});
    //update
    for (auto stone : group->stones){
        groups.at(stone) = group;
    }
}

bool Board::is_capture(Move move){
    std::unordered_set<Point> neighbors = adjacency_table->at({move.x, move.y});
    for (auto neighbor: neighbors){
        if (groups.count(neighbor)){
            std::shared_ptr<Group> group = groups.at(neighbor);
            if (group->color != move.turn){
                if (group->number_of_liberties() == 1){
                    return true;
                }
            }
        }
    }
    return false;
}

bool Board::is_self_capture(Move move){
    std::unordered_set<Point> neighbors = adjacency_table->at({move.x, move.y});
    std::unordered_set<Point> s = {{move.x, move.y}};
    std::unordered_set<Point> l = adjacency_table->at({move.x, move.y});
    std::shared_ptr<Group> group = std::make_shared<Group>(move.turn,s,l);

    for (auto neighbor: neighbors){
        if (groups.count(neighbor)){
            group->remove_liberty(neighbor);
            std::shared_ptr<Group> neighbour_group = groups.at(neighbor);
            if(neighbour_group->color == move.turn){
                group->merge_group(*neighbour_group);
            }
        }
    }
    if(group->number_of_liberties() == 0){
        return true;
    }
    else{
        return false;
    }
}

std::vector<Point> Board::legal_moves(int color){
    std::vector<Point> moves;
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            Move move(color,i,j);
            if (is_legal(move)){
                moves.push_back({i,j});
            }
        }
    }
    return moves;
}

void Board::print_board(){
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            if(groups.count({i,j})){
                if(groups.at({i,j})->color == 1){
                    printf("B");
                }
                else{
                    printf("W");
                }
            }
            else{
                printf(".");
            }
        }
        printf("\n");
    }  
}



void Board::create_adjacency_table(){
    std::cout<<"Creating adjacency table\n";
    // Inside points
    for (int i = 1; i < size-1; i++){
        for (int j = 1; j < size-1; j++){
            adjacency_table->insert({{i,j},{{i-1, j}, {i+1, j}, {i, j-1}, {i, j+1}}});
        }
    }
    
    // Edge points
    for (int i = 1; i < size-1; i++){
        adjacency_table->insert({{0, i},{{1, i}, {0, i-1}, {0, i+1}}});
        adjacency_table->insert({{size-1, i},{{size-2, i}, {size-1, i-1}, {size-1, i+1}}});
        adjacency_table->insert({{i, 0},{{i, 1}, {i-1, 0}, {i+1, 0}}});
        adjacency_table->insert({{i, size-1},{{i, size-2}, {i-1, size-1}, {i+1, size-1}}});
    }
    
    // Corner points
    adjacency_table->insert({{0, 0},{{0, 1}, {1, 0}}});
    adjacency_table->insert({{0, size-1},{{1, size-1}, {0, size-2}}});
    adjacency_table->insert({{size-1, size-1},{{size-2, size-1}, {size-1, size-2}}});
    adjacency_table->insert({{size-1, 0},{{size-1, 1}, {size-2, 0}}});
}