#include <cstdio>
#include "engine.h"
#include "gameplay/gamestate.h"
#include "agents/mcts.h"
#include <thread>
#include <atomic>
#include "network_evaluation.h"

std::atomic<int> bwins;
std::atomic<int> wwins;


int play_mcts_game(MCTS agent){
    Board board(9);
    GameState gamestate(1,9,board);
    while(true){
        Point move = agent.select_move(gamestate);
        if(gamestate.board.is_capture(Move(gamestate.turn,move.first,move.second))){
            break;
        }
        gamestate.play_move(move);
        //no more legal moves, count as loss
        if(gamestate.legal_moves().empty()){
            printf("Player %d passes\n",gamestate.turn);
            break;
        }
    }
    return gamestate.turn;
}


void selfplay(int n,torch::jit::script::Module model,int n_threads){
    int size = 9;
    ExtendedEncoder encoder(size);
    NetworkEvaluation evaluator(size,model,encoder);
    MCTS agent(size,evaluator);
    play_mcts_games(n,agent);
}

void play_mcts_games(int n,MCTS agent){
    int temp_bwins = 0;
    int temp_wwins = 0;
    //start agents
    for(int i = 0; i < n; i++){
        int result = play_mcts_game(agent);
        std::cout<<"Completed game "<<i<<"\n";
        if(result == 1){
            temp_bwins++;
        }else if(result == 2){
            temp_wwins++;
        }
    }
    std::cout<<"Black wins: "<<temp_bwins<<"\n";
    std::cout<<"White wins: "<<temp_wwins<<"\n";
}




int play_game(RandomAgent agent,std::shared_ptr<AdjacencyTable> adjacency_table){
    Board board(9,adjacency_table);
    GameState gamestate(1,9,board);
    
    while(true){
        Point move = agent.select_move(gamestate);
        if(gamestate.board.is_capture(Move(gamestate.turn,move.first,move.second))){
            break;
        }
        gamestate.play_move(move);
        //no more legal moves, count as loss
        if(gamestate.legal_moves().empty()){
            printf("Player %d passes\n",gamestate.turn);
            break;
        }
        
    }
    //gamestate.print_gamestate();
    return gamestate.turn;
}

void play_random_games(int n){
    int temp_bwins = 0;
    int temp_wwins = 0;
    Board b(9);
    std::shared_ptr<AdjacencyTable> adjacency_table = b.adjacency_table;
    RandomAgent agent(9);
    //start agents
    for(int i = 0; i < n; i++){
        int result = play_game(agent,adjacency_table);
        if(result == 1){
            temp_bwins++;
        }else if(result == 2){
            temp_wwins++;
        }
    }
    std::atomic_fetch_add(&bwins,temp_bwins);
    std::atomic_fetch_add(&wwins,temp_wwins);
}

void play_games(int n,int n_threads){
    int block_size = n/n_threads;
    std::vector<std::thread> threads(n_threads);

    //start agents
    for(int i = 0; i < n_threads; i++){
        threads[i] = std::thread(play_random_games,block_size);
    }


    for(int i = 0; i < n_threads; i++){
        threads[i].join();
    }
    printf("Black wins: %d\n",bwins.load());
    printf("White wins: %d\n",wwins.load());
}