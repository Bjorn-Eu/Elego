#include "extended_encoder.h"
#include <torch/torch.h>

torch::Tensor ExtendedEncoder::encode(GameState gamestate){
    torch::Tensor black = torch::zeros({1,1,size, size}); //current player
    torch::Tensor white = torch::zeros({1,1,size, size}); //opponent
    torch::Tensor one = torch::zeros({1,1,size, size});
    torch::Tensor two = torch::zeros({1,1,size, size});
    torch::Tensor three = torch::zeros({1,1,size, size});

    
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            Point point(i,j);
            if(gamestate.board.groups.count(point)){
                if(gamestate.board.groups.at(point)->color == gamestate.turn){
                    black[0][0][i][j] = 1;
                }else{
                    white[0][0][i][j] = 1;
                }
                if(gamestate.board.groups.at(point)->number_of_liberties() == 1){
                    one[0][0][i][j] = 1;
                }
                else if(gamestate.board.groups.at(point)->number_of_liberties() == 2){
                    two[0][0][i][j] = 1;
                }
                else if(gamestate.board.groups.at(point)->number_of_liberties() == 3){
                    three[0][0][i][j] = 1;
                }
            }
        }
    }

    return torch::cat({black,white,one,two,three},1);
}