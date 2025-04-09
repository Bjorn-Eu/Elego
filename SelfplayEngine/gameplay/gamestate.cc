#include "gamestate.h"

void GameState::play_move(Point point){
    Move move = Move(turn,point.first,point.second);
    if(board.is_capture(move)){
        is_over = true;
    }
    board.play_move(move);
    turn = (turn == 1) ? 2 : 1;
    move_history.push_back(move);
}

void GameState::print_gamestate(){
    board.print_board();
}

bool GameState::is_legal(Move move){
    if(move.turn != turn){
        return false;
    }
    return board.is_legal(move);
}

std::vector<Point> GameState::legal_moves(){
    return board.legal_moves(turn);
}

bool GameState::is_terminal(){
    return is_over;
}