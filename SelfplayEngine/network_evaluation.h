#ifndef NETWORK_EVALUATION_H
#define NETWORK_EVALUATION_H
#include <torch/torch.h>
#include <torch/script.h>
#include <tuple>
#include <map>
#include "gameplay/gamestate.h"
#include "encoders/extended_encoder.h"

class NetworkEvaluation{
    public:
        int size;
        torch::jit::script::Module module;
        ExtendedEncoder encoder;

        NetworkEvaluation(int s, torch::jit::script::Module m,ExtendedEncoder e) : size(s), module(m), encoder(e){}
        std::tuple<double,std::map<Point,double>> evaluate2(GameState gamestate);
        std::tuple<double,std::map<Point,double>> evaluate(GameState gamestate);
};



#endif 