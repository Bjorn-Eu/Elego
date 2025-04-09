#ifndef RANDOM_AGENT_H
#define RANDOM_AGENT_H
#include "../gameplay/gamestate.h"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>


class RandomAgent{
    private:
        int size;
    public:
        RandomAgent(int s): size(s){       
            srand(time(NULL));
        }
        Point select_move(GameState gamestate);
};
#endif