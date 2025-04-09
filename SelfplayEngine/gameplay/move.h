#ifndef MOVE_H
#define MOVE_H


class Move{
    public:
        int turn;
        int x;
        int y;
        
        Move(int m_turn, int m_x,int m_y):turn(m_turn), x(m_x), y(m_y) {}
        void print_move();
};


#endif