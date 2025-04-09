#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <string>
#include <unistd.h>
#include "engine.h"



int main(int argc,char* argv[]){
  //read input args
  int opt;
  char* filename; 
  int n_games = 100; //default
  int n_threads = 1;
  while((opt = getopt(argc,argv,"t:n:")) != -1){
      switch(opt){
          case 't':
              n_threads = atoi(optarg);
              break;
          case 'n':
              n_games = atoi(optarg);
              break;
          default:
              std::cerr<<"Usage: "<<argv[0]<<" [-n Number of games] [-t number of threads] model\n";
              return EXIT_FAILURE;

      }
  }

  if(argc == (optind+1)){
      filename = argv[optind];
  } else{
      std::cerr<<"Usage: "<<argv[0]<<" [-n Number of games] [-t number of threads] model\n";
      return EXIT_FAILURE;
  }

  std::string model_path = filename;

  //attempt to load model
  torch::jit::script::Module model;
  try {
    model = torch::jit::load(model_path);
  }
  catch (const c10::Error& e) {
    std::cerr << "Error loading the model\n";
    return EXIT_FAILURE;
  }
  std::cout<<"Succesfully loaded model\n";
  std::cout<<"Running "<<n_games<<" selfplay games with network "<<model_path<< " using "<<n_threads<<" threads\n";
  selfplay(n_games,model,n_threads);
  
  //play random games
  //play_games(n_games*1000,n_threads);

  return EXIT_SUCCESS;
}