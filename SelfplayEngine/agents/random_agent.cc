#include "random_agent.h"
#include <vector>
#include <cstdio>

Point RandomAgent::select_move(GameState gamestate){
    std::vector<Point> legal_moves = gamestate.legal_moves();
    int random_index = rand()%(legal_moves.size());
    Point random_move = legal_moves[random_index];
    
    return random_move;
}