#include "write_sgf.h"
#include <iostream>
#include <fstream>
void write_sgf(){
    std::ofstream sgf_file("game.sgf");
    if (sgf_file.is_open()){
        sgf_file << "(;GM[1]FF[4]CA[UTF-8]AP[MyGo:0.1]KM[6.5]SZ[19]\n";
        sgf_file << "PB[BlackPlayer]PW[WhitePlayer]\n";
        sgf_file << "RE[W+0.5]\n";
        sgf_file << ";\n";
        sgf_file << ")\n";
        sgf_file.close();
    }
    else{
        std::cerr << "Unable to open file";
    }
}