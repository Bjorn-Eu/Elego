#ifndef MCTS_H
#define MCTS_H

#include "../gameplay/gamestate.h"
#include "../encoders/extended_encoder.h"
#include "node.h"
#include <functional>
#include "../network_evaluation.h"

class MCTS{
    int size;
    NetworkEvaluation evaluator;
    int playouts = 100;

    public:
        MCTS(int s,NetworkEvaluation e):size(s), evaluator(e) {}
        Point select_move(GameState gamestate);
        void expand_node(std::shared_ptr<Node> node);
};


#endif