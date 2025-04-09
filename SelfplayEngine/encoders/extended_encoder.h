#ifndef EXTENDED_ENCODER_H
#define EXTENDED_ENCODER_H
#include <torch/torch.h>
#include "../gameplay/gamestate.h"

class ExtendedEncoder {
    public:
        int size;
        ExtendedEncoder(int s):size(s){};
        torch::Tensor encode(GameState gamestate);
};


#endif