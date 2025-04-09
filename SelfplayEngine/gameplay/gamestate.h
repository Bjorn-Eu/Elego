#ifndef GAMESTATE_H
#define GAMESTATE_H
#include "move.h"
#include "board.h"
#include <vector>
#include <memory>




class GameState{
    private:

    public:
        int turn;
        int size;
        Board board;
        GameState(int t,int s,Board b):turn(t), size(s), board(b){}
        void play_move(Point move);
        void print_gamestate();
        bool is_legal(Move move);
        std::vector<Point> legal_moves();
        bool is_over = false;
        bool is_terminal();
};


#endif