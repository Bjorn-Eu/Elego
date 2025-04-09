#ifndef ENGINE_H
#define ENGINE_H
#include <torch/torch.h>
#include <torch/script.h>
#include "agents/random_agent.h"
#include "agents/mcts.h"



//return winner
int play_game(RandomAgent agent,std::shared_ptr<AdjacencyTable> adjacency_table);

int play_mcts_game(MCTS agent);
void play_random_games(int n);

void play_mcts_games(int n,MCTS agent);
void play_games(int n,int n_threads);

void selfplay(int n,torch::jit::script::Module model,int n_threads);



#endif