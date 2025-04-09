#include "write_sgf.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

void write_sgf(GameState gamestate){
    std::ofstream sgf_file("games.sgf",std::ios::app);
    if (sgf_file.is_open()){

        sgf_file << "(;GM[1]FF[4]SZ["<<gamestate.size<<"]\n";
        sgf_file << "PB[BlackPlayer]PW[WhitePlayer]\n";
        if(gamestate.turn == 1){
            sgf_file << "RE[W]\n";
        }
        else{
            sgf_file << "RE[B]\n";
        }
        
        for(auto move : gamestate.move_history){
            if(move.turn == 1){
                sgf_file<<"B";
            }
            else{
                sgf_file<<"W";
            }
            sgf_file<<"["<<(char)(move.y+97)<<(char)(move.x+97)<<"];";
        }
        sgf_file << ";\n";
        sgf_file << ")\n";
        sgf_file.close();
    }
    else{
        std::cerr << "Unable to open file";
    }
}