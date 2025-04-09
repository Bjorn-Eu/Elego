#ifndef POLICY_H
#define POLICY_H
#include "torch/torch.h"
#include <torch/script.h> 
#include "../gameplay/gamestate.h"
#include "../encoders/extended_encoder.h"
class PolicyAgent{
    public:
        int size;
        torch::jit::script::Module module;
        ExtendedEncoder encoder;

        PolicyAgent(int s, torch::jit::script::Module m,ExtendedEncoder e) : size(s), module(m), encoder(e){}
        Point select_move(GameState gamestate);
};

#endif