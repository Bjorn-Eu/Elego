
#include <cstdio>
#include "policy_agent.h"
#include "random_agent.h"

Point PolicyAgent::select_move(GameState gamestate){
    torch::Tensor input = encoder.encode(gamestate);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
    module.eval();
    torch::NoGradGuard no_grad;
    auto output = module.forward(inputs);
    at::Tensor value = output.toTuple()->elements()[0].toTensor();
    at::Tensor policy = output.toTuple()->elements()[1].toTensor();
    auto out_index = (policy.argmax()).item<int>();

    int y = 1+out_index/9;
    int x = out_index%9; 
    Point point = {x,y};
    if(gamestate.is_legal(Move(gamestate.turn,x,y))){
        return point;
    }
    else{ //if selected move is illegal select random
        RandomAgent random_agent(9);
        return random_agent.select_move(gamestate);
    }
}