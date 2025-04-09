#include "network_evaluation.h"
#include <tuple>
#include "agents/test_policy.h"


std::tuple<double,std::map<Point,double>> NetworkEvaluation::evaluate(GameState gamestate){
    torch::Tensor input = encoder.encode(gamestate);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
    module.eval();
    torch::NoGradGuard no_grad;
    auto output = module.forward(inputs);
    at::Tensor value = output.toTuple()->elements()[0].toTensor();
    at::Tensor policy = output.toTuple()->elements()[1].toTensor();
    auto np_policy = policy.data_ptr<float>();
    std::map<Point,double> policy_map;
    for (int i = 0; i < size; i++){
        for (int j = 0; j < size; j++){
            policy_map.insert({{i,j},np_policy[i*size+j]});
        }
    }
    return std::make_tuple(value.item<double>(),policy_map);
}

std::tuple<double,std::map<Point,double>> NetworkEvaluation::evaluate2(GameState gamestate){
    auto x = encoder.encode(gamestate);
    return get_adv_out(x,size);
}